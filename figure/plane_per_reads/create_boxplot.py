#!/usr/bin/env python3
"""
Box plot script for reads count data.
Creates box plots where each reads_count column has its own box.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ISCA-style plot configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['patch.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5

# Configuration
SAMPLE_NUM = 50  # Number of columns to display (0 to SAMPLE_NUM-1)

def load_data(filename):
    """Load the reorganized CSV data."""
    return pd.read_csv(filename)

def create_boxplot(df, figsize=(4.5, 3)):
    """
    Create box plots for reads count data.
    
    Args:
        df: DataFrame with reorganized data
        figsize: Figure size tuple
    """
    # Get reads count columns
    reads_columns = [col for col in df.columns if col.startswith('reads_count_')]
    
    # Select head 0 from every 4th layer (L0H0, L4H0, L8H0, etc.)
    selected_columns = []
    for col in reads_columns:
        x = int(col.split('_')[-1])  # Extract the number from reads_count_x
        layer = x // 8  # 8 heads per layer
        if layer > 31:  # Limit to first step for clarity
            continue

        head = x % 8
        # Select only head 0 from layers 0, 4, 8, 12, 16, 20, 24, 28, etc.
        if head == 0 and layer % 3 == 0:
            selected_columns.append(col)
        # if head == 0 and layer % 4 == 2:
        #     selected_columns.append(col)
    
    reads_columns = selected_columns
    
    # Calculate max value for normalization from ALL reads_count columns (not just selected)
    # This ensures consistent normalization regardless of layer selection
    all_reads_columns = [col for col in df.columns if col.startswith('reads_count_')]
    max_value = 0
    for col in all_reads_columns:
        values = df[col].values
        max_value = max(max_value, np.max(values))
    max_value = max_value *0.77
    print(f"Global max value for normalization: {max_value}")
    
    # Prepare data for box plot
    data_for_boxplot = []
    labels = []
    
    for col in reads_columns:
        # Extract the head index from column name
        x = int(col.split('_')[-1])
        layer = x // 8  # 8 heads per layer
        head = x % 8
        label = f'{layer}'
        # label = f'L{layer}H{head}'
        
        # Get all values for this column (across all plane groups)
        values = df[col].values
        
        data_for_boxplot.append(values)
        labels.append(label)
    
    # Normalize all data by the maximum value
    data_for_boxplot_normalized = [data / max_value for data in data_for_boxplot]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # ISCA-style color (colorblind-friendly)
    color_box = "#FFC000"
    # color_box = "#2C5AA0"
    # color_box = "#83A6E2"
    
    # Create box plot with normalized data
    box_plot = plt.boxplot(data_for_boxplot_normalized, 
                          labels=labels,
                          patch_artist=True,
                          showfliers=False,  # Don't show outliers as dots
                          notch=False,
                          widths=0.6)
    
    # Customize box plot colors with ISCA style
    for patch in box_plot['boxes']:
        patch.set_facecolor(color_box)
        patch.set_alpha(0.9)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    
    # Style the whiskers, caps, and medians
    for whisker in box_plot['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.2)
    
    for cap in box_plot['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.2)
    
    for median in box_plot['medians']:
        median.set_color("red")
        median.set_linewidth(1.5)
    
    # Customize the plot with ISCA style
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Norm. Reads per Plane', fontsize=12)
    
    # Rotate x-axis labels for better readability and center align
    plt.xticks(rotation=0, ha='center')
    
    # Increase font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
    
    # Add grid for better readability - set zorder to draw below box plot
    ax.set_axisbelow(True)  # This makes grid lines appear below the plot elements
    ax.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    
    # Set y-axis to show normalized scale (0 to 1)
    ax.set_ylim(0, 1.0)
    
    # Set spine linewidths for ISCA style
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Show only every nth label to avoid overcrowding
    if len(labels) > 20:
        tick_freq = len(labels) // 20
        for i, label in enumerate(ax.get_xticklabels()):
            if i % tick_freq != 0:
                label.set_visible(False)
    
    plt.tight_layout()
    return plt.gcf()

def main():
    """Main function to create box plot visualizations."""
    
    # Load data
    print("Loading reorganized data...")
    df = load_data("data/reads_per_head_reorganized_Meta-Llama-3.1-8B_32800_budget0.10_replica4.csv")
    print(f"Data shape: {df.shape}")
    print(f"Plane groups: {df['plane_group'].min()} to {df['plane_group'].max()}")
    
    # Get number of reads_count columns
    reads_columns = [col for col in df.columns if col.startswith('reads_count_')]
    print(f"Total reads_count columns: {len(reads_columns)}")
    
    # Count how many columns will be selected (head 0 from every 4th layer)
    selected_count = 0
    for col in reads_columns:
        x = int(col.split('_')[-1])
        layer = x // 8  # 8 heads per layer
        head = x % 8
        if head == 0 and layer % 4 == 0:
            selected_count += 1
    
    print(f"Selecting head 0 from every 4th layer: {selected_count} columns")
    print("Selected: L0H0, L4H0, L8H0, L12H0, ...")
    
    # Create box plot with selected columns
    print("Creating box plot...")
    fig_box = create_boxplot(df)
    plt.savefig("reads_boxplot.png", dpi=300, bbox_inches='tight')
    plt.savefig("reads_boxplot.pdf", bbox_inches='tight')
    print("Saved: reads_boxplot.png")
    print("Saved: reads_boxplot.pdf")
    plt.show()
    
if __name__ == "__main__":
    main()