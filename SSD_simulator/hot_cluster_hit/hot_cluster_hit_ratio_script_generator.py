#!/usr/bin/env python
import itertools
import os
import glob

# Base profiling directory
PROFILING_BASE_DIR = "/home/juchanlee/MagicDec/profile/data/"

# Define fixed options and their possible values for hot cluster hit ratio analysis
fixed_option_values = {
    "--page_size_bytes": [16384],
    "--vector_bytes": [4],
    "--num_heads": [8],
    "--cluster_size": [32],
    "--layer_num": [48],
    "--head_dim": [128],
    "--dram_capacity_gb": [128],
    "--profiling_dir": [PROFILING_BASE_DIR],
}

# Define variable options and their possible values
option_values = {
    "--model_name": ["qwen2.5-14b"],
    "--dataset": ["pg19"],
    "--prefix_len": ["16416"],
    "--batch_size": ["4", "16", "64", "128"],
    # "--prefix_len": ["8224", "16416", "32800"],
}

# Boolean flags (hot cluster hit ratio analysis doesn't need many flags)
store_true_flags = ["--constrained"]

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
    return "_".join(parts)

# Build a command line for hot_cluster_hit_ratio.py
def build_command(var_keys, var_tuple, flags, generate_name, model_name, dataset, prefix_len, folder_type):
    cmd = ["python hot_cluster_hit_ratio.py"]
        
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
        budget_ratio = "0.25"
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

def main():
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
    
    filtered_folders = []
    for folder in data_folders:
        model_dataset_dir = folder['model_dataset_dir']
        # Check if this folder matches any of our target prefix lengths
        for target_prefix_len in option_values["--prefix_len"]:
            target_dir_pattern = f"{target_model_name}_{target_dataset}_{target_prefix_len}"
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
                break
    
    if not filtered_folders:
        print(f"No matching folders found for model {target_model_name}, dataset {target_dataset}!")
        return
    
    print(f"\nFiltered to {len(filtered_folders)} matching folders")
    
    # Generate all combinations of variable options
    keys, values = zip(*option_values.items())
    combinations = list(itertools.product(*values))
    
    # Build the full list of commands
    generated_scripts = []
    for combo in combinations:
        # Extract values from the combination
        combo_dict = dict(zip(keys, combo))
        
        # Only process folders that match the current combination's model/dataset/prefix_len
        matching_folders = [
            folder for folder in filtered_folders
            if (folder['model_name'] == combo_dict["--model_name"] and
                folder['dataset'] == combo_dict["--dataset"] and
                folder['prefix_len'] == combo_dict["--prefix_len"])
        ]
        
        for flag_mask in itertools.product([False, True], repeat=len(store_true_flags)):
            active_flags = [flag for flag, use in zip(store_true_flags, flag_mask) if use]
            
            # Generate commands for each matching data folder
            for folder in matching_folders:
                cmd_script = build_command(
                    keys, combo, active_flags, 
                    folder['generate_name'],
                    folder['model_name'],
                    folder['dataset'],
                    folder['prefix_len'],
                    folder['type']  # Pass the folder type (speculate or verify)
                )
                generated_scripts.append(cmd_script)
    
    # Assemble script content
    template = "#!/bin/bash\n\n" + "\n\necho \"Processing next hot cluster hit ratio configuration...\"\n\n".join(generated_scripts)
    
    # Write to a file named for the analysis type
    script_name = make_script_filename(option_values)
    with open(script_name + ".sh", "w") as f:
        f.write(template)
    
    print(f"\nGenerated {script_name}.sh with {len(generated_scripts)} commands.")
    print(f"This will analyze hot cluster hit ratios for:")
    print(f"  - {len(option_values['--prefix_len'])} prefix lengths: {option_values['--prefix_len']}")
    print(f"  - {len(set(f['generate_name'] for f in filtered_folders))} unique generate patterns")
    print(f"  - {2**len(store_true_flags)} flag combinations each")
    
    # Print summary of what will be analyzed
    unique_patterns = set()
    for folder in filtered_folders:
        unique_patterns.add(f"{folder['generate_name']} ({folder['type']})")
    
    print(f"\nGenerate patterns to be analyzed:")
    for pattern in sorted(unique_patterns):
        print(f"  - {pattern}")

if __name__ == "__main__":
    main()
