import itertools

# Define options and their possible values
option_values = {
    # fixed
    "--num_channels": [1],
    "--chips_per_channel": [1],
    "--dies_per_chip": [1],
    "--page_size_bytes": [4096],
    "--vector_bytes": [4],
    "--flash_read_latency_us": [50],
    "--num_heads": [8],
    "--cluster_size": [16],
    "--window_size": [16],

    # variables
    "--hot_cluster_ratio": [0.01,0.02,0.04,0.08],
    # "--planes_per_die": [64, 32, 16, 8],
    "--planes_per_die": [32],
    "--profiling_dir": [
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.25KV_16385_clustersize_16",
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.12KV_16385_clustersize_16",
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.06KV_16385_clustersize_16",
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.03KV_16385_clustersize_16",
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.25KV_32769_clustersize_16",
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.12KV_32769_clustersize_16",
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.06KV_32769_clustersize_16",
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.03KV_32769_clustersize_16",
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.25KV_65537_clustersize_16",
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.12KV_65537_clustersize_16",
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.06KV_65537_clustersize_16",
        "/home/juchanlee/MagicDec/Engine/RetrievalAttention/profile/data/data_superclustersize_4_0.03KV_65537_clustersize_16"
    ],

}

# Boolean flags
# store_true_flags = ["--hot_cluster_duplicate", "--hotness_aware_layout"]
store_true_flags = ["--hot_cluster_duplicate"]

# Generate all combinations of non-boolean options
keys, values = zip(*option_values.items())
combinations = list(itertools.product(*values))

# Function to build a command from parameters and flags

def build_command(param_tuple, flags):
    cmd = ["python simulator.py"]
    cmd.append("--max_latency_calculate") #always append this flag
    for key, val in zip(keys, param_tuple):
        cmd.append(f"{key} {val}")
    for flag in flags:
        cmd.append(flag)
    return " \
".join(cmd)

# Example: generate commands for all combinations, toggling boolean flags

generated_scripts = []
for combo in combinations:
    # You can choose to include or exclude each store_true flag
    for flag_mask in itertools.product([False, True], repeat=len(store_true_flags)):
        active_flags = [flag for flag, use in zip(store_true_flags, flag_mask) if use]
        cmd_script = build_command(combo, active_flags)
        generated_scripts.append(cmd_script)

# Save to file
template = "#!/bin/bash\n\n" + "\n\n".join(generated_scripts)
with open("run_simulations.sh", "w") as f:
    f.write(template)

print("Generated run_simulations.sh with", len(generated_scripts), "commands.")
