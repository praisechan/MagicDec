#!/usr/bin/env python3
"""
Script to generate energy breakdown stacked bar graph from CSV data.
Creates a stacked bar chart showing energy consumption breakdown by component.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set font to DejaVu Sans (clean sans-serif font similar to Arial)
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_and_process_data(csv_path):
    """Load CSV data and process it for plotting."""
    df = pd.read_csv(csv_path, index_col=0)
    return df

def create_energy_breakdown_figure(csv_path, output_path=None):
    """Create the energy breakdown stacked bar graph figure."""
    # Load data
    df = load_and_process_data(csv_path)
    
    # Set up the figure - academic style
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    
    # Academic color palette - professional and colorblind-friendly
    # Grayscale to blue gradient matching throughput figure style
    colors = {
        'Cell read': '#2C5AA0',      # Light gray
        'I/O': '#4A90E2',             # Medium gray
        'Compute (In-Flash)': "#9FC9FA",  # Dark gray
        'DRAM': '#E8E8E8'             # Professional blue
    }
    
    # Get the systems (rows) and components (columns)
    systems = df.index.tolist()
    components = df.columns.tolist()
    
    # Width of bars
    bar_width = 0.4
    x_positions = np.arange(len(systems))
    
    # Edge properties for bars - academic style
    edge_linewidth = 0.8
    edge_color = '#333333'
    
    # Create stacked bars
    bottom = np.zeros(len(systems))
    bars = []
    
    for component in components:
        values = df[component].values
        # Add diagonal pattern for DRAM and Compute (In-Flash)
        if component == 'DRAM':
            bar = ax.bar(x_positions, values, bar_width, bottom=bottom, 
                        label=component, color=colors[component], alpha=0.9,
                        hatch='///', edgecolor=edge_color, linewidth=edge_linewidth)
        elif component == 'Compute (In-Flash)':
            bar = ax.bar(x_positions, values, bar_width, bottom=bottom, 
                        label=component, color=colors[component], alpha=0.9,
                        hatch='...', edgecolor=edge_color, linewidth=edge_linewidth)
        else:
            bar = ax.bar(x_positions, values, bar_width, bottom=bottom, 
                        label=component, color=colors[component], alpha=0.9,
                        edgecolor=edge_color, linewidth=edge_linewidth)
        bars.append(bar)
        bottom += values
    
    # Customize plot - academic style
    ax.set_ylabel('Energy Consumption (Norm.)', fontsize=11, fontweight='normal')
    # ax.set_xlabel('System Configuration', fontsize=11, fontweight='normal')
    
    # Set x-axis labels - academic style
    ax.set_xticks(x_positions)
    ax.set_xticklabels(systems, fontsize=11, rotation=15, ha='right')
    
    # Set y-axis tick fontsize - academic style
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=11)
    
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
              ncol=4, fontsize=10, frameon=False, columnspacing=1.0, 
              handletextpad=0.5, handlelength=1.5)
    
    # Adjust layout to prevent overlap - academic spacing
    plt.tight_layout()
    
    # Save the figure - academic quality
    if output_path is None:
        output_path = Path(csv_path).parent / 'energy_breakdown_stacked.png'
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.show()
    
    print(f"Figure saved to: {output_path}")

def main():
    """Main function to run the script."""
    # Path to the CSV file
    csv_file = "energy_calculation.CSV"
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"Error: CSV file not found at {csv_file}")
        print("Please make sure the file exists or update the path.")
        return
    
    # Create the figure
    create_energy_breakdown_figure(csv_file)

if __name__ == "__main__":
    main()
