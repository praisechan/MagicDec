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

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Configuration
SAMPLE_NUM = 50  # Number of columns to display (0 to SAMPLE_NUM-1)

def load_data(filename):
    """Load the reorganized CSV data."""
    return pd.read_csv(filename)

def create_boxplot(df, figsize=(8, 6)):
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
        if layer > 33:  # Limit to first step for clarity
            continue

        head = x % 8
        # Select only head 0 from layers 0, 4, 8, 12, 16, 20, 24, 28, etc.
        if head == 0 and layer % 4 == 1:
            selected_columns.append(col)
    
    reads_columns = selected_columns
    
    # Prepare data for box plot
    data_for_boxplot = []
    labels = []
    
    for col in reads_columns:
        # Extract the head index from column name
        x = int(col.split('_')[-1])
        layer = x // 8  # 8 heads per layer
        head = x % 8
        label = f'{layer+1}'
        # label = f'L{layer}H{head}'
        
        # Get all values for this column (across all plane groups)
        values = df[col].values
        
        data_for_boxplot.append(values)
        labels.append(label)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Create box plot
    box_plot = plt.boxplot(data_for_boxplot, 
                          labels=labels,
                          patch_artist=True,
                          showfliers=False,  # Don't show outliers as dots
                          notch=False)
    
    # Customize box plot colors
    # colors = plt.cm.viridis(np.linspace(0, 1, len(data_for_boxplot)))
    colors = ["#FFC000"] * len(data_for_boxplot)  # Uniform color
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize the plot
    plt.xlabel('Layer', fontsize=24)
    plt.ylabel('Num. Reads per Plane', fontsize=24)
    # plt.title('Distribution of Reads Count Across Plane Groups for Each Attention Head', fontsize=14, pad=20)
    
    # Rotate x-axis labels for better readability and center align
    plt.xticks(rotation=0, ha='center')
    
    # Increase font size of tick labels
    plt.tick_params(axis='both', which='major', labelsize=24)
    
    # Add grid for better readability - set zorder to draw below box plot
    ax = plt.gca()
    ax.set_axisbelow(True)  # This makes grid lines appear below the plot elements
    plt.grid(True, alpha=0.5, axis='y')
    
    # Control ytick frequency separately
    # Get current y-axis limits
    y_min, y_max = ax.get_ylim()
    # Set custom ytick frequency (adjust the step value as needed)
    ytick_step = 2  # Change this value to control frequency
    yticks = np.arange(0, y_max + ytick_step, ytick_step)
    ax.set_yticks(yticks)
    
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
    plt.show()
    
if __name__ == "__main__":
    main()