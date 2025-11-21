#!/usr/bin/env python3
"""
Script to generate throughput bar graphs from CSV data.
Creates three subplots (one for each model) with grouped bars by batch size.
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
    batch_sizes = ['batch8', 'batch16', 'batch32', 'batch64', 'batch128']
    
    # Set up the figure with subplots - academic style
    fig, axes = plt.subplots(1, 3, figsize=(18, 2.7))  # Wider to accommodate 5 batches
    
    # Academic color palette - professional and colorblind-friendly
    # Based on scientific publication standards (blues, grays, accent colors)
    colors = [
        # '#E8E8E8',  # Light gray for SSD(w/o SD) and SSD(w/ SD)
        '#E8E8E8',  # Same gray for SSD pair
        '#A8A8A8',  # Medium gray for DRAM+SSD
        '#6B6B6B',  # Dark gray for Inf.DRAM
        '#4A90E2',  # Professional blue for InstAttention
        '#2C5AA0',  # Deep blue for FlashSpec and FlashSpec++
        '#2C5AA0'   # Same deep blue for FlashSpec pair
    ]
    
    # Width of bars and positions - academic style
    bar_width = 0.18  # Adjusted for 5 batches
    x_positions = np.arange(len(batch_sizes))
    
    # Edge properties for bars - academic style
    edge_linewidth = 0.8  # Thinner, more subtle edges
    edge_color = '#333333'  # Dark gray instead of pure black
    
    # Create subplots for each model
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = df[df['model'] == model]
        
        # Plot bars for each case
        for j, case in enumerate(cases):
            case_data = model_data[model_data['case'] == case]
            if not case_data.empty:
                values = [case_data[batch].iloc[0] for batch in batch_sizes]
                x_pos = x_positions + j * bar_width
                
                # Use same color for FlashSpec and FlashSpec++ but different patterns
                if case == 'FlashSpec++':
                    # Use the same color as FlashSpec but with subtle hatching pattern
                    flashspec_idx = list(cases).index('FlashSpec') if 'FlashSpec' in cases else j
                    bars = ax.bar(x_pos, values, bar_width, label=case, color=colors[flashspec_idx], 
                                alpha=0.9, hatch='///', edgecolor=edge_color, linewidth=edge_linewidth)
                elif case == 'SSD(w/o SD)':
                    pass
                    # Use the same color as SSD(w/ SD) but with subtle hatching pattern
                    # ssd_with_idx = list(cases).index('SSD(w/ SD)') if 'SSD(w/ SD)' in cases else j
                    # bars = ax.bar(x_pos, values, bar_width, label=case, color=colors[ssd_with_idx], 
                    #             alpha=0.9, hatch='...', edgecolor=edge_color, linewidth=edge_linewidth)
                else:
                    bars = ax.bar(x_pos, values, bar_width, label=case, color=colors[j], alpha=0.9,
                                edgecolor=edge_color, linewidth=edge_linewidth)
                
                # Add value labels on top of bars (optional - uncomment if desired)
                # for k, bar in enumerate(bars):
                #     height = bar.get_height()
                #     ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                #             f'{values[k]:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Customize subplot - academic style
        # ax.set_title(f'{model}', fontsize=15, pad=0)
        if i == 0:
            ax.set_ylabel('Throughput (token/s)', fontsize=13, fontweight='normal')
        
        # Set x-axis labels - academic style
        ax.set_xticks(x_positions + bar_width * (len(cases) - 1) / 2)
        ax.set_xticklabels(['8', '16', '32', '64', '128'], fontsize=13)
        ax.set_xlabel('Batch Size', fontsize=13, fontweight='normal')

        # Set y-axis tick fontsize - academic style
        ax.tick_params(axis='y', labelsize=13)
        ax.tick_params(axis='x', labelsize=13)
        
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
        
        # Add title inside the plot at upper center
        ax.text(0.5, 0.95, f'{model}', transform=ax.transAxes, 
                fontsize=15, ha='center', va='top')
    
    # Create a single legend above all subplots - academic style
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
               ncol=7, fontsize=13, frameon=False, columnspacing=2, handletextpad=0.5)
    
    # Adjust layout to prevent overlap - academic spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.98, wspace=0.15)
    
    # Save the figure - academic quality
    if output_path is None:
        output_path = Path(csv_path).parent / 'throughput_comparison_more_batch.png'
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.show()
    
    print(f"Figure saved to: {output_path}")
    
    # Create a second figure with log-scale y-axis
    fig_log, axes_log = plt.subplots(1, 3, figsize=(18, 3.5))
    
    # Create subplots for each model with log scale
    for i, model in enumerate(models):
        ax = axes_log[i]
        model_data = df[df['model'] == model]
        
        # Plot bars for each case
        for j, case in enumerate(cases):
            case_data = model_data[model_data['case'] == case]
            if not case_data.empty:
                values = [case_data[batch].iloc[0] for batch in batch_sizes]
                x_pos = x_positions + j * bar_width
                
                # Use same color for FlashSpec and FlashSpec++ but different patterns
                if case == 'FlashSpec++':
                    flashspec_idx = list(cases).index('FlashSpec') if 'FlashSpec' in cases else j
                    bars = ax.bar(x_pos, values, bar_width, label=case, color=colors[flashspec_idx], 
                                alpha=0.9, hatch='///', edgecolor=edge_color, linewidth=edge_linewidth)
                elif case == 'SSD(w/o SD)':
                    pass
                else:
                    bars = ax.bar(x_pos, values, bar_width, label=case, color=colors[j], alpha=0.9,
                                edgecolor=edge_color, linewidth=edge_linewidth)
        
        # Customize subplot - academic style with log scale
        # ax.set_title(f'{model}', fontsize=15, pad=0)
        if i == 0:
            ax.set_ylabel('Throughput (tokens/s, log scale)', fontsize=13, fontweight='normal')
        
        # Set x-axis labels - academic style
        ax.set_xticks(x_positions + bar_width * (len(cases) - 1) / 2)
        ax.set_xticklabels(['8', '16', '32', '64', '128'], fontsize=13)
        ax.set_xlabel('Batch Size', fontsize=13, fontweight='normal')

        # Set y-axis tick fontsize - academic style
        ax.tick_params(axis='y', labelsize=13)
        ax.tick_params(axis='x', labelsize=13)
        
        # Set y-axis to log scale
        ax.set_yscale('log')
        
        # Academic-style grid - more subtle
        ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5, which='both')
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
        
        # Add title inside the plot at upper center
        ax.text(0.5, 0.95, f'{model}', transform=ax.transAxes, 
                fontsize=15, ha='center', va='top')
    
    # Create a single legend above all subplots - academic style
    handles, labels = axes_log[0].get_legend_handles_labels()
    fig_log.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), 
                   ncol=7, fontsize=13, frameon=False, columnspacing=2, handletextpad=0.5)
    
    # Adjust layout to prevent overlap - academic spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.15, left=0.08, right=0.98, wspace=0.15)
    
    # Save the log-scale figure
    output_path_log = Path(csv_path).parent / 'throughput_comparison_more_batch_log.png'
    plt.savefig(output_path_log, dpi=600, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.show()
    
    print(f"Log-scale figure saved to: {output_path_log}")

def main():
    """Main function to run the script."""
    # Path to the CSV file
    csv_file = "data/main_figure_more_batch.csv"
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"Error: CSV file not found at {csv_file}")
        print("Please make sure the file exists or update the path.")
        return
    
    # Create the figure
    create_throughput_figure(csv_file)

if __name__ == "__main__":
    main()