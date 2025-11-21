#!/usr/bin/env python3
"""
Script to generate throughput bar graphs from CSV data (by sequence length).
Creates a bar graph with grouped bars by sequence length.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set font to DejaVu Sans (clean sans-serif font similar to Arial)
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_and_process_data(csv_path):
    """Load CSV data and process it for plotting."""
    df = pd.read_csv(csv_path)
    
    # Forward fill the model names (they're only in the first row of each group)
    df['model'] = df['model'].fillna(method='ffill')
    
    return df

def create_throughput_figure(csv_path, output_path=None):
    """Create the throughput bar graph figure."""
    # Load data
    df = load_and_process_data(csv_path)
    
    # Get unique models and cases
    models = df['model'].unique()
    cases = df['case'].unique()
    seqlens = ['8K', '16K', '24K', '32K', '40K']
    
    # Set up the figure with single plot - academic style
    fig, ax = plt.subplots(1, 1, figsize=(6, 2.7))  # Single model plot
    
    # Academic color palette - professional and colorblind-friendly
    # Based on scientific publication standards (blues, grays, accent colors)
    colors = [
        # '#E8E8E8',  # Light gray for SSD(w/o SD) and SSD(w/ SD)
        '#E8E8E8',  # Same gray for SSD pair
        '#A8A8A8',  # Medium gray for DRAM+SSD
        '#6B6B6B',  # Dark gray for Inf.DRAM
        '#4A90E2',  # Professional blue for InstAttention
        '#2C5AA0',  # Deep blue for Flash-PIM and FlashSpec
        '#2C5AA0'   # Same deep blue for FlashSpec
    ]
    
    # Width of bars and positions - academic style
    bar_width = 0.13  # Increased width to reduce gaps
    x_positions = np.arange(len(seqlens))
    
    # Edge properties for bars - academic style
    edge_linewidth = 0.8  # Thinner, more subtle edges
    edge_color = '#333333'  # Dark gray instead of pure black
    
    # Use the first (and only) model
    model = models[0]
    model_data = df[df['model'] == model]
    
    # Plot bars for each case
    for j, case in enumerate(cases):
        case_data = model_data[model_data['case'] == case]
        if not case_data.empty:
            values = [case_data[seqlen].iloc[0] for seqlen in seqlens]
            x_pos = x_positions + j * bar_width
            
            # Use same color for Flash-PIM and FlashSpec but different patterns
            if case == 'FlashSpec++':
                # Use the same color as Flash-PIM but with subtle hatching pattern
                flash_pim_idx = list(cases).index('FlashSpec') if 'FlashSpec' in cases else j
                bars = ax.bar(x_pos, values, bar_width, label=case, color=colors[flash_pim_idx], 
                            alpha=0.9, hatch='///', edgecolor=edge_color, linewidth=edge_linewidth)
            elif case == 'SSD(w/o SD)':
                # Use the same color as SSD(w/ SD) but with subtle hatching pattern
                ssd_with_idx = list(cases).index('SSD(w/ SD)') if 'SSD(w/ SD)' in cases else j
                bars = ax.bar(x_pos, values, bar_width, label=case, color=colors[ssd_with_idx], 
                            alpha=0.9, hatch='...', edgecolor=edge_color, linewidth=edge_linewidth)
            else:
                bars = ax.bar(x_pos, values, bar_width, label=case, color=colors[j], alpha=0.9,
                            edgecolor=edge_color, linewidth=edge_linewidth)
    
    # Customize plot - academic style
    # ax.set_title(f'{model}', fontsize=15, pad=10)
    ax.set_ylabel('Throughput (Normalized)', fontsize=10, fontweight='normal')
    
    # Set x-axis labels - academic style
    ax.set_xticks(x_positions + bar_width * (len(cases) - 1) / 2)
    ax.set_xticklabels(['8K', '16K', '24K', '32K', '40K'], fontsize=11)
    ax.set_xlabel('Sequence Length', fontsize=11, fontweight='normal')
    
    # Set y-axis tick fontsize - academic style
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)

    # Academic-style grid - more subtle
    ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Show all 4 spines (borders) with proper styling
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_linewidth(0.8)
    ax.spines['right'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Create legend above the plot - academic style
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.40), 
              ncol=3, fontsize=10, frameon=False, columnspacing=0.3, handletextpad=0.5, handlelength=1.0)
    
    # Adjust layout to prevent overlap - academic spacing
    plt.tight_layout()
    
    # Save the figure - academic quality
    if output_path is None:
        output_path = Path(csv_path).parent / 'seqlen_throughput_comparison_qwen14b_bsz32_new.png'
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.show()
    
    print(f"Figure saved to: {output_path}")

def main():
    """Main function to run the script."""
    # Path to the CSV file
    csv_file = "data/seqlen_throughput_qwen14b_normalized.CSV"
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"Error: CSV file not found at {csv_file}")
        print("Please make sure the file exists or update the path.")
        return
    
    # Create the figure
    create_throughput_figure(csv_file)

if __name__ == "__main__":
    main()
