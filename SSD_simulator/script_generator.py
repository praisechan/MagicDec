import itertools

# Define fixed options and their possible values
fixed_option_values = {
    "--num_channels": [1],
    "--chips_per_channel": [1],
    "--dies_per_chip": [1],
    "--page_size_bytes": [4096],
    "--vector_bytes": [4],
    "--flash_read_latency_us": [50],
    "--num_heads": [8],
    "--cluster_size": [16],
    "--window_size": [64],
    "--profiling_dir": [
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/",
    ],
}

# Define variable options and their possible values
option_values = {
    "--num_replica": [1,2,4,8],
    # "--num_replica": [8],
    # "--prefix_len": [8193, 16385, 32769, 65537],
    "--prefix_len": [8193],
    "--budget_ratio": [0.12],
    "--hot_cluster_ratio": [0.01, 0.02, 0.04, 0.08, 0.16],
    "--planes_per_die": [32],
}

# Boolean flags
store_true_flags = ["--hot_cluster_duplicate"]

# Generate filename based only on single-valued variable options
def make_script_filename(var_vals):
    parts = []
    for opt, vals in var_vals.items():
        if len(vals) == 1:
            clean = opt.lstrip('-').replace('-', '_')
            parts.append(f"{clean}_{vals[0]}")
    return "_".join(parts)

# Build a command line including fixed options, variable options, and flags
def build_command(var_keys, var_tuple, flags):
    cmd = ["python simulator.py"]
    cmd.append("--max_latency_calculate")
    script_name = make_script_filename(option_values)
    cmd.append(f"--csv_path {script_name}")
    # include all fixed options
    for key, vals in fixed_option_values.items():
        cmd.append(f"{key} {vals[0]}")
    # include variable options for this combo
    for key, val in zip(var_keys, var_tuple):
        cmd.append(f"{key} {val}")
    # include any store_true flags
    for flag in flags:
        cmd.append(flag)
    # join with backslashes for multiline readability
    return " \
".join(cmd)

# Generate all combinations of variable options
keys, values = zip(*option_values.items())
combinations = list(itertools.product(*values))

# Build the full list of commands
generated_scripts = []
for combo in combinations:
    for flag_mask in itertools.product([False, True], repeat=len(store_true_flags)):
        active_flags = [flag for flag, use in zip(store_true_flags, flag_mask) if use]
        cmd_script = build_command(keys, combo, active_flags)
        generated_scripts.append(cmd_script)

# Assemble script content
template = "#!/bin/bash\n\n" + "\n\n".join(generated_scripts)


# Write to a file named for the single-valued variable options only
script_name = make_script_filename(option_values)
with open(script_name + ".sh", "w") as f:
    f.write(template)

print(f"Generated {script_name} with {len(generated_scripts)} commands.")
