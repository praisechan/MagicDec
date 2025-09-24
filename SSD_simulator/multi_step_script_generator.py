import itertools
import os
import glob
import argparse

# Base profiling directory
# PROFILING_BASE_DIR = "/home/juchanlee/MagicDec/profile/data_kl_conf_lowhigh_optimized_cluster32_gamma28_for_SSDsim/"
PROFILING_BASE_DIR = "/home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32"
# Default value - can be overridden by command line argument
# PROFILING_BASE_DIR = "/home/juchanlee/MagicDec/profile/data_0.02/"

ADDITIONAL_COMMENT = ""
# ADDITIONAL_COMMENT = "for_0.1"

# Define fixed options and their possible values
fixed_option_values = {
    "--num_channels": [1],
    "--chips_per_channel": [1],
    "--dies_per_chip": [1],
    "--page_size_bytes": [16384],
    "--vector_bytes": [4],
    "--flash_read_latency_us": [50],
    "--num_heads": [8],
    "--cluster_size": [32],
    "--window_size": [64],
    "--layer_num": [32],
    "--profiling_dir": [
        PROFILING_BASE_DIR
    ],
}

# Define variable options and their possible values
option_values = {
    "--num_replica": [4],
    "--prefix_len": ["8224"],
    "--hot_cluster_ratio": [0.08],
    "--planes_per_die": [32],
    "--model_name": ["Meta-Llama-3.1-8B"],
    "--dataset": ["pg19"],
}

# Boolean flags
store_true_flags = ["--hot_cluster_duplicate"]

def sort_directories_numerically(dirs):
    """Sort directories with pattern 'type_X_Y' in numerical order."""
    def extract_numbers(dir_path):
        dir_name = os.path.basename(dir_path)
        # Extract numbers from patterns like 'speculate_X_Y' or 'verify_X_Y'
        parts = dir_name.split('_')
        if len(parts) >= 3:
            try:
                # Return (first_number, second_number) for sorting
                return (int(parts[1]), int(parts[2]))
            except ValueError:
                # If parsing fails, fall back to string sorting
                return (float('inf'), float('inf'))
        return (float('inf'), float('inf'))
    
    return sorted(dirs, key=extract_numbers)

def discover_data_folders():
    """Discover all available data folders in the profiling directory."""
    data_folders = []
    
    # Look for model_dataset_prefix folders
    model_dataset_dirs = glob.glob(os.path.join(PROFILING_BASE_DIR, "*_*_*"))
    for model_dataset_dir in model_dataset_dirs:
        if os.path.isdir(model_dataset_dir):
            dir_name = os.path.basename(model_dataset_dir)
            
            # Find speculate and verify folders
            speculate_dirs = glob.glob(os.path.join(model_dataset_dir, "speculate_*"))
            verify_dirs = glob.glob(os.path.join(model_dataset_dir, "verify_*"))
            
            # Sort directories numerically
            speculate_dirs = sort_directories_numerically(speculate_dirs)
            verify_dirs = sort_directories_numerically(verify_dirs)
            
            for spec_dir in speculate_dirs:
                if os.path.isdir(spec_dir):
                    spec_name = os.path.basename(spec_dir)
                    data_folders.append({
                        'model_dataset_dir': dir_name,
                        'generate_name': spec_name,
                        'type': 'speculate'
                    })
            
            for ver_dir in verify_dirs:
                if os.path.isdir(ver_dir):
                    ver_name = os.path.basename(ver_dir)
                    data_folders.append({
                        'model_dataset_dir': dir_name,
                        'generate_name': ver_name,
                        'type': 'verify'
                    })
    
    return data_folders

# Generate filename based only on single-valued variable options
def make_script_filename(var_vals):
    parts = []
    for opt, vals in var_vals.items():
        if len(vals) == 1:
            clean = opt.lstrip('-').replace('-', '_')
            parts.append(f"{clean}_{vals[0]}")
    return "_".join(parts)+f"_{ADDITIONAL_COMMENT}"

# Build a command line including fixed options, variable options, and flags
def build_command(var_keys, var_tuple, flags, generate_name, model_name, dataset, prefix_len, folder_type, verify_budget_ratio=0.25):
    cmd = ["python simulator.py"]
    cmd.append("--max_latency_calculate")
    script_name = make_script_filename(option_values)
    cmd.append(f"--csv_path {script_name}")
    
    # include all fixed options
    for key, vals in fixed_option_values.items():
        cmd.append(f"{key} {vals[0]}")
    
    # include variable options for this combo (excluding model_name, dataset, prefix_len, budget_ratio)
    for key, val in zip(var_keys, var_tuple):
        if key not in ["--model_name", "--dataset", "--prefix_len", "--budget_ratio"]:
            cmd.append(f"{key} {val}")
    
    # Set budget_ratio based on folder type
    if folder_type == "verify":
        budget_ratio = str(verify_budget_ratio)
        # budget_ratio = "0.10"
    else:  # speculate
        budget_ratio = "0.02"
    
    cmd.append(f"--budget_ratio {budget_ratio}")
    
    # Add the specific data folder arguments
    cmd.append(f"--model_name {model_name}")
    cmd.append(f"--dataset {dataset}")
    cmd.append(f"--generate_name {generate_name}")
    cmd.append(f"--prefix_len {prefix_len}")
    
    # include any store_true flags
    for flag in flags:
        cmd.append(flag)
    
    # join with backslashes for multiline readability
    return " \\\n".join(cmd)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate multi-step SSD simulator scripts')
    parser.add_argument('--profiling_base_dir', 
                        type=str,
                        default="/home/juchanlee/MagicDec/profile/data/",
                        help='Base directory for profiling data (default: /home/juchanlee/MagicDec/profile/data/)')
    parser.add_argument('--verify_budget_ratio',
                        type=float,
                        default=0.25,
                        help='Budget ratio for verify folder type (default: 0.25)')
    return parser.parse_args()

def main():
    global PROFILING_BASE_DIR
    
    # Parse command line arguments
    args = parse_arguments()
    # PROFILING_BASE_DIR = args.profiling_base_dir
    
    # Update the fixed_option_values with the new profiling directory
    fixed_option_values["--profiling_dir"] = [PROFILING_BASE_DIR]
    
    print(f"Using profiling base directory: {PROFILING_BASE_DIR}")
    
    # Discover all available data folders
    data_folders = discover_data_folders()
    
    if not data_folders:
        print("No data folders found!")
        return
    
    print(f"Found {len(data_folders)} data folders to process:")
    for folder in data_folders[:10]:  # Show first 10 as example
        print(f"  {folder['model_dataset_dir']}/{folder['generate_name']} ({folder['type']})")
    if len(data_folders) > 10:
        print(f"  ... and {len(data_folders) - 10} more")
    
    # Filter to specific model and dataset based on option values
    target_model_name = option_values["--model_name"][0]
    target_dataset = option_values["--dataset"][0] 
    target_prefix_len = option_values["--prefix_len"][0]
    target_dir_pattern = f"{target_model_name}_{target_dataset}_{target_prefix_len}"
    
    filtered_folders = []
    for folder in data_folders:
        model_dataset_dir = folder['model_dataset_dir']
        if target_dir_pattern in model_dataset_dir:
            # Extract model, dataset, prefix_len from the directory name
            parts = model_dataset_dir.split('_')
            if len(parts) >= 3:
                model_name = parts[0]  # e.g., qwen2.5-14b
                dataset = parts[1]  # e.g., pg19
                prefix_len = parts[2]  # e.g., 8224
                
                filtered_folders.append({
                    'generate_name': folder['generate_name'],
                    'type': folder['type'],
                    'model_name': model_name,
                    'dataset': dataset,
                    'prefix_len': prefix_len
                })
    
    if not filtered_folders:
        print(f"No {target_dir_pattern} folders found!")
        return
    
    print(f"\nFiltered to {len(filtered_folders)} {target_dir_pattern} folders")
    
    # Generate all combinations of variable options
    keys, values = zip(*option_values.items())
    combinations = list(itertools.product(*values))
    
    # Build the full list of commands
    generated_scripts = []
    for combo in combinations:
        for flag_mask in itertools.product([False, True], repeat=len(store_true_flags)):
            active_flags = [flag for flag, use in zip(store_true_flags, flag_mask) if use]
            
            # Generate commands for each data folder
            for folder in filtered_folders:
                cmd_script = build_command(
                    keys, combo, active_flags, 
                    folder['generate_name'],
                    folder['model_name'],
                    folder['dataset'],
                    folder['prefix_len'],
                    folder['type'],  # Pass the folder type (speculate or verify)
                    args.verify_budget_ratio  # Pass the verify budget ratio
                )
                generated_scripts.append(cmd_script)
    
    # Assemble script content
    template = "#!/bin/bash\n\n" + "\n\necho \"Processing next configuration...\"\n\n".join(generated_scripts)
    
    # Write to a file named for the single-valued variable options only
    script_name = make_script_filename(option_values) + "_multi_step"
    with open(script_name + ".sh", "w") as f:
        f.write(template)
    
    print(f"\nGenerated {script_name}.sh with {len(generated_scripts)} commands.")
    print(f"This will process {len(filtered_folders)} data folders with {len(combinations)} parameter combinations and {2**len(store_true_flags)} flag combinations each.")

if __name__ == "__main__":
    main()
