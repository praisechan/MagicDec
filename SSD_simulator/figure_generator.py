import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('convert_to_figure.csv')

# Clean up column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Forward fill the prefix_len values
df['prefix_len'] = df['prefix_len'].fillna(method='ffill')

# Get unique prefix lengths
prefix_lengths = df['prefix_len'].unique()

# Define colors for each prefix length
colors = ['blue', 'red', 'green', 'orange']

# Create individual plots for each prefix length
for i, prefix_len in enumerate(prefix_lengths):
    plt.figure(figsize=(6, 6))
    
    # Filter data for current prefix length
    subset = df[df['prefix_len'] == prefix_len].copy()
    
    color = colors[i % len(colors)]
    
    # Plot duplicate speedup (solid line)
    plt.plot(subset['hot_cluster_ratio'], subset['duplicate speedup'], 
             color=color, linestyle='-', marker='o', linewidth=2,
             label=f'Duplicate Speedup')
    
    # Plot overlap+duplicate speedup (dashed line, same color)
    plt.plot(subset['hot_cluster_ratio'], subset['overlap+dupliacate speedup'], 
             color=color, linestyle='--', marker='s', linewidth=2,
             label=f'Overlap+Duplicate Speedup')
    
    # Plot overlap speedup as horizontal line (where data exists)
    overlap_values = subset['overlap speedup'].dropna()
    if not overlap_values.empty:
        overlap_value = overlap_values.iloc[0]  # Take first non-NaN value
        plt.axhline(y=overlap_value, color='gray', linestyle=':', alpha=0.8, linewidth=2,
                   label=f'Overlap Speedup')
    
    # Plot ideal speedup as horizontal line (where data exists)
    ideal_values = subset['ideal speedup'].dropna()
    if not ideal_values.empty:
        ideal_value = ideal_values.iloc[0]  # Take first non-NaN value
        plt.axhline(y=ideal_value, color='black', linestyle='-', alpha=0.6, linewidth=3,
                   label=f'Ideal Speedup')

    # Set x-axis to log scale
    plt.xscale('log')
    
    # Set custom x-axis ticks to show only the hot cluster ratio values
    plt.xticks(subset['hot_cluster_ratio'], subset['hot_cluster_ratio'])

    # Customize the plot
    plt.xlabel('Hot Cluster Ratio', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title(f'Speedup vs Hot Cluster Ratio (Prefix Length = {prefix_len})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    filename = f'speedup_prefix_len_{prefix_len}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    
    # Close the figure to free memory
    plt.close()

# Also create a combined plot and save it
plt.figure(figsize=(12, 8))

for i, prefix_len in enumerate(prefix_lengths):
    # Filter data for current prefix length
    subset = df[df['prefix_len'] == prefix_len].copy()
    
    color = colors[i % len(colors)]
    
    # Plot duplicate speedup (solid line)
    plt.plot(subset['hot_cluster_ratio'], subset['duplicate speedup'], 
             color=color, linestyle='-', marker='o', 
             label=f'Duplicate Speedup (prefix_len={prefix_len})')
    
    # Plot overlap+duplicate speedup (dotted line, same color)
    plt.plot(subset['hot_cluster_ratio'], subset['overlap+dupliacate speedup'], 
             color=color, linestyle='--', marker='s', 
             label=f'Overlap+Duplicate Speedup (prefix_len={prefix_len})')
    
    # Plot overlap speedup as horizontal line (where data exists)
    overlap_values = subset['overlap speedup'].dropna()
    if not overlap_values.empty:
        overlap_value = overlap_values.iloc[0]  # Take first non-NaN value
        plt.axhline(y=overlap_value, color=color, linestyle=':', alpha=0.7,
                   label=f'Overlap Speedup (prefix_len={prefix_len})')
    
    # Plot ideal speedup as horizontal line (where data exists)
    ideal_values = subset['ideal speedup'].dropna()
    if not ideal_values.empty:
        ideal_value = ideal_values.iloc[0]  # Take first non-NaN value
        plt.axhline(y=ideal_value, color=color, linestyle='-', alpha=0.5, linewidth=3,
                   label=f'Ideal Speedup (prefix_len={prefix_len})')

# Set x-axis to log scale
plt.xscale('log')

# Set custom x-axis ticks to show only the hot cluster ratio values
all_ratios = sorted(df['hot_cluster_ratio'].unique())
plt.xticks(all_ratios, all_ratios)

# Customize the combined plot
plt.xlabel('Hot Cluster Ratio', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.title('Speedup vs Hot Cluster Ratio for Different Prefix Lengths', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Save the combined figure
plt.savefig('speedup_comparison_combined.png', dpi=300, bbox_inches='tight')
print("Saved: speedup_comparison_combined.png")

# Show the combined plot
plt.show()