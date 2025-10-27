#!/usr/bin/env python3
"""
Comprehensive plotting script for reorganized plane reads data.
Creates various types of visualizations based on the reorganized CSV data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_data(filename):
    """Load the reorganized CSV data."""
    return pd.read_csv(filename)

def create_heatmap(df, title="Reads Count Heatmap", figsize=(16, 8), sample_cols=None):
    """
    Create a heatmap showing reads count across plane groups and quotients.
    
    Args:
        df: DataFrame with reorganized data
        title: Title for the plot
        figsize: Figure size tuple
        sample_cols: Number of columns to sample (None for all)
    """
    # Prepare data for heatmap
    reads_columns = [col for col in df.columns if col.startswith('reads_count_')]
    
    if sample_cols:
        # Select first sample_cols columns and discard the rest
        reads_columns = reads_columns[:sample_cols]
    
    heatmap_data = df[['plane_group'] + reads_columns].set_index('plane_group')
    
    # Create custom x-tick labels for the sampled columns
    custom_labels = []
    for col in reads_columns:
        x = int(col.split('_')[-1])  # Extract the number from reads_count_x
        layer = x // 8
        head = x % 8
        # custom_labels.append(f'L{layer}')
        custom_labels.append(f'L{layer}H{head}')
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    # Determine tick label frequency
    if sample_cols and len(custom_labels) > 20:
        tick_freq = 8  # Show about 20 labels
        tick_labels = [custom_labels[i] if i % tick_freq == 0 else '' for i in range(len(custom_labels))]
    else:
        tick_labels = custom_labels
    
    sns.heatmap(heatmap_data, 
                cmap='viridis', 
                cbar_kws={'label': 'Number of reads per plane'},
                xticklabels=tick_labels,
                yticklabels=4
                )

    # Reverse y-axis order
    plt.gca().invert_yaxis()

    # Increase font size of tick labels
    plt.tick_params(axis='both', which='major', labelsize=24)

    # Increase colorbar font sizes
    cbar = plt.gcf().axes[-1]  # Get the colorbar axis
    cbar.tick_params(labelsize=24)  # Colorbar tick labels
    cbar.set_ylabel('Number of reads', fontsize=24)  # Colorbar label
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=0, ha='left')
    
    # plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel('Attention Head Index (Layer-Head)', fontsize=24)
    plt.ylabel('Plane Index', fontsize=24)
    plt.tight_layout()
    
    return plt.gcf()

def create_minmax_plot(df, title="Reads Count Min/Max by Attention Head", figsize=(8, 6), window_size=8):

    reads_columns = [col for col in df.columns if col.startswith('reads_count_')]
    
    # Calculate statistics for each reads_count column (across all plane groups)
    stats_data = []
    for col in reads_columns:
        reads_index = int(col.split('_')[-1])
        values = df[col].values
        
        stats_data.append({
            'reads_count_index': reads_index,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Apply smoothing with rolling window
    stats_df['min_smooth'] = stats_df['min'].rolling(window=window_size, center=True, min_periods=1).mean()
    stats_df['max_smooth'] = stats_df['max'].rolling(window=window_size, center=True, min_periods=1).mean()
    stats_df['median_smooth'] = stats_df['median'].rolling(window=window_size, center=True, min_periods=1).mean()
    stats_df['q25_smooth'] = stats_df['q25'].rolling(window=window_size, center=True, min_periods=1).mean()
    stats_df['q75_smooth'] = stats_df['q75'].rolling(window=window_size, center=True, min_periods=1).mean()
    
    # Normalize all values to 10 as 1 (divide by 10)
    stats_df['min_smooth_norm'] = stats_df['min_smooth'] / 10
    stats_df['max_smooth_norm'] = stats_df['max_smooth'] / 10
    stats_df['median_smooth_norm'] = stats_df['median_smooth'] / 10

    fig, ax = plt.subplots(figsize=figsize)
    
    # # Add fill_between for min-max range
    # ax.fill_between(stats_df['reads_count_index'], 
    #                 stats_df['min_smooth_norm'], 
    #                 stats_df['max_smooth_norm'], 
    #                 alpha=0.2, color="#86aeff")
    
    # Plot min and max as separate lines
    ax.plot(stats_df['reads_count_index'], stats_df['max_smooth_norm'], 
            color='#2F5597', linewidth=2.5, label='Max', linestyle='-')
    
    ax.plot(stats_df['reads_count_index'], stats_df['min_smooth_norm'], 
            color="#BD3C00", linewidth=2.5, label='Min', linestyle='-')
    
    # # Plot Q1 and Q3 as separate lines
    # ax.plot(stats_df['reads_count_index'], stats_df['q75_smooth'], 
    #         color='#e74c3c', linewidth=2.0, label='Q₃ (75th percentile)', 
    #         linestyle='--', alpha=0.7)
    
    # ax.plot(stats_df['reads_count_index'], stats_df['q25_smooth'], 
    #         color='#3498db', linewidth=2.0, label='Q₁ (25th percentile)', 
    #         linestyle='--', alpha=0.7)
    
    # # Plot median line
    # ax.plot(stats_df['reads_count_index'], stats_df['median_smooth_norm'], 
    #         color='#2c3e50', linewidth=2.5, label='Median', zorder=10)

    # # Add horizontal line at y=0.5 for ideal value
    # ax.axhline(y=0.5, color="#007918", linewidth=3.5, linestyle='--', 
    #            label='Ideal', zorder=5, alpha=0.8)
    
    # Add subtle grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    ax.set_xlabel('Attention Head ID', fontsize=24)
    ax.set_ylabel('Norm. Reads per Plane', fontsize=24)
    ax.tick_params(axis='both', labelsize=24)
    
    # Remove space between graph and side edges
    ax.set_xlim(stats_df['reads_count_index'].min(), stats_df['reads_count_index'].max())
        
    # Add legend outside the plot area (below the graph)
    ax.legend(fontsize=22, loc='upper center', ncol=4, framealpha=0.0,
              bbox_to_anchor=(0.5, 1.15))
    
    plt.tight_layout()
    return fig

def create_correlation_heatmap(df, title="Plane Group Correlation", figsize=(12, 10), sample_size=100):
    """
    Create correlation heatmap between plane groups based on reads patterns.
    """
    reads_columns = [col for col in df.columns if col.startswith('reads_count_')]
    
    # Sample columns for correlation analysis
    if len(reads_columns) > sample_size:
        step = len(reads_columns) // sample_size
        sampled_columns = reads_columns[::step][:sample_size]
    else:
        sampled_columns = reads_columns
    
    # Prepare data for correlation
    correlation_data = df[['plane_group'] + sampled_columns].set_index('plane_group')
    
    # Calculate correlation matrix
    correlation_matrix = correlation_data.T.corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Plane Group', fontsize=12)
    plt.ylabel('Plane Group', fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()

def create_temporal_pattern(df, title="Temporal Reads Pattern", figsize=(16, 8)):
    """
    Create a visualization showing reads pattern over quotients (time-like dimension).
    """
    reads_columns = [col for col in df.columns if col.startswith('reads_count_')]
    quotients = [int(col.split('_')[-1]) for col in reads_columns]
    
    # Calculate statistics across all plane groups for each quotient
    quotient_stats = []
    for col in reads_columns:
        quotient_idx = int(col.split('_')[-1])
        values = df[col].values
        quotient_stats.append({
            'quotient': quotient_idx,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        })
    
    quotient_df = pd.DataFrame(quotient_stats)
    
    plt.figure(figsize=figsize)
    
    # Plot mean with error bars
    plt.fill_between(quotient_df['quotient'], 
                     quotient_df['mean'] - quotient_df['std'],
                     quotient_df['mean'] + quotient_df['std'],
                     alpha=0.3, color='blue', label='±1 std')
    
    plt.plot(quotient_df['quotient'], quotient_df['mean'], 
             'b-', linewidth=2, label='Mean')
    
    plt.fill_between(quotient_df['quotient'], 
                     quotient_df['min'], 
                     quotient_df['max'],
                     alpha=0.2, color='red', label='Min-Max Range')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Quotient Index', fontsize=12)
    plt.ylabel('Reads Count', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def main():
    """Main function to create all visualizations."""
    
    # Load data
    print("Loading reorganized data...")
    df = load_data("data/reads_per_head_reorganized_Meta-Llama-3.1-8B_32800_budget0.10_replica4.csv")
    print(f"Data shape: {df.shape}")
    print(f"Plane groups: {df['plane_group'].min()} to {df['plane_group'].max()}")
    
    # Create visualizations
    figures = {}
    
    print("Creating heatmap...")
    figures['heatmap'] = create_heatmap(df, 
                                        sample_cols=64,  # Sample 96 columns for visibility
                                       title="Reads Count Heatmap (Sampled)")
    plt.savefig("reads_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Creating Min/Max plot...")
    figures['minmax'] = create_minmax_plot(df)
    plt.savefig("reads_minmax.png", dpi=300, bbox_inches='tight')
    plt.show()

    # print("Creating line plot...")
    # figures['lineplot'] = create_line_plot(df, 
    #                                       sample_planes=8,  # Sample 8 plane groups
    #                                       title="Reads Count Trends (Selected Planes)")
    # plt.savefig("reads_lineplot.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    # print("Creating statistics plot...")
    # figures['statistics'] = create_statistics_plot(df)
    # plt.savefig("reads_statistics.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    # print("Creating correlation heatmap...")
    # figures['correlation'] = create_correlation_heatmap(df, sample_size=50)
    # plt.savefig("reads_correlation.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    # print("Creating temporal pattern...")
    # figures['temporal'] = create_temporal_pattern(df)
    # plt.savefig("reads_temporal.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("All plots created successfully!")
    print("Saved files:")
    print("- reads_heatmap.png")
    print("- reads_lineplot.png") 
    print("- reads_statistics.png")
    print("- reads_correlation.png")
    print("- reads_temporal.png")
    
    return figures

if __name__ == "__main__":
    main()