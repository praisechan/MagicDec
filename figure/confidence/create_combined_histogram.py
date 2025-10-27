#!/usr/bin/env python3
"""
Create combined histogram visualization for draft and reject confidence data
Generates a single figure with dual y-axes: one for draft tokens and one for reject tokens
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_combined_histogram(csv_path, output_path=None, figsize=(12, 8)):
    """
    Create a combined histogram plot with draft and reject data on dual y-axes.
    
    Args:
        csv_path: Path to the CSV file containing histogram data
        output_path: Path to save the output figure (optional)
        figsize: Figure size tuple
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Find draft and reject rows
    draft_rows = df[df['experiment'].str.contains('draft')]
    reject_rows = df[df['experiment'].str.contains('reject') & ~df['experiment'].str.contains('budget')]
    
    if draft_rows.empty or reject_rows.empty:
        raise ValueError("Could not find both draft and reject data in the CSV file")
    
    # Use the first draft and reject rows
    draft_row = draft_rows.iloc[0]
    reject_row = reject_rows.iloc[0]
    
    # Get the bin columns (exclude 'experiment' column)
    bin_columns = [col for col in df.columns if col != 'experiment']
    
    # Extract counts for draft and reject
    draft_counts = draft_row[bin_columns].values
    reject_counts = reject_row[bin_columns].values
    
    # Convert bin labels to start values and combine into 0.1-width bins
    bin_starts = []
    for bin_label in bin_columns:
        # Extract start value from "0.00-0.02" format
        start, end = map(float, bin_label.split('-'))
        bin_starts.append(start)
    
    bin_starts = np.array(bin_starts)
    
    # Create new bins with 0.1 width (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
    new_bin_edges = np.arange(0.0, 1.1, 0.1)
    new_bin_centers = new_bin_edges[:-1] + 0.05  # Centers at 0.05, 0.15, ..., 0.95
    
    # Combine counts into new bins
    new_draft_counts = np.zeros(len(new_bin_centers))
    new_reject_counts = np.zeros(len(new_bin_centers))
    
    for i, (draft_count, reject_count, bin_start) in enumerate(zip(draft_counts, reject_counts, bin_starts)):
        # Find which new bin this old bin belongs to
        bin_idx = int(bin_start * 10)  # 0.00->0, 0.02->0, ..., 0.98->9
        if bin_idx < len(new_bin_centers):
            new_draft_counts[bin_idx] += draft_count
            new_reject_counts[bin_idx] += reject_count
    
    # Use the new combined bins
    bin_centers = new_bin_centers
    draft_counts = new_draft_counts
    reject_counts = new_reject_counts
    
    # Calculate percentages
    draft_total = np.sum(draft_counts)
    reject_total = np.sum(reject_counts)
    
    draft_percentages = (draft_counts / draft_total) * 100
    reject_percentages = (reject_counts / reject_total) * 100
    
    # Create the figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Calculate bin width for bar plots (now using 0.1 width bins)
    bin_width = 0.1
    
    # Plot draft tokens on primary y-axis (left)
    color1 = 'steelblue'
    ax1.bar(bin_centers - bin_width/4, draft_percentages, width=bin_width*0.4, 
            alpha=0.7, color=color1, edgecolor='black', linewidth=0.5, label='Draft Tokens')
    ax1.set_xlabel('Confidence', fontsize=24)
    ax1.set_ylabel('Draft Tokens (%)', color=color1, fontsize=24)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis for reject tokens
    ax2 = ax1.twinx()
    color2 = 'darkred'
    ax2.bar(bin_centers + bin_width/4, reject_percentages, width=bin_width*0.4, 
            alpha=0.7, color=color2, edgecolor='black', linewidth=0.5, label='Rejected Tokens')
    ax2.set_ylabel('Rejected Tokens (%)', color=color2, fontsize=24)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Set x-axis properties
    # Create custom tick positions for 0.2 width intervals
    # Ticks at 0.2, 0.4, 0.6, 0.8 positioned between pairs of bins
    custom_tick_positions = [0.2, 0.4, 0.6, 0.8]
    custom_tick_labels = ['0.2', '0.4', '0.6', '0.8']
    
    ax1.set_xticks(custom_tick_positions)
    ax1.set_xticklabels(custom_tick_labels)
    ax1.set_xlim(-0.05, 1.05)
    
    # Set y-axis limits
    ax1.set_ylim(0, 35)
    ax2.set_ylim(0, 35)
    
    # Add title and legend
    plt.title('Draft vs Rejected Tokens Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Total draft tokens: {draft_total}")
    print(f"Total reject tokens: {reject_total}")
    print(f"Draft acceptance rate: {(draft_total / (draft_total + reject_total)) * 100:.2f}%")
    print(f"Rejection rate: {(reject_total / (draft_total + reject_total)) * 100:.2f}%")
    
    return fig, ax1, ax2

def create_stacked_histogram(csv_path, output_path=None, figsize=(8, 6)):
    """
    Create a stacked histogram plot showing draft and reject data in the same bars.
    
    Args:
        csv_path: Path to the CSV file containing histogram data
        output_path: Path to save the output figure (optional)
        figsize: Figure size tuple
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Find draft and reject rows
    draft_rows = df[df['experiment'].str.contains('draft')]
    reject_rows = df[df['experiment'].str.contains('reject') & ~df['experiment'].str.contains('budget')]
    
    if draft_rows.empty or reject_rows.empty:
        raise ValueError("Could not find both draft and reject data in the CSV file")
    
    # Use the first draft and reject rows
    draft_row = draft_rows.iloc[0]
    reject_row = reject_rows.iloc[0]
    
    # Get the bin columns (exclude 'experiment' column)
    bin_columns = [col for col in df.columns if col != 'experiment']
    
    # Extract counts for draft and reject
    draft_counts = draft_row[bin_columns].values
    reject_counts = reject_row[bin_columns].values
    
    # Convert bin labels to start values and combine into 0.1-width bins
    bin_starts = []
    for bin_label in bin_columns:
        # Extract start value from "0.00-0.02" format
        start, end = map(float, bin_label.split('-'))
        bin_starts.append(start)
    
    bin_starts = np.array(bin_starts)
    
    # Create new bins with 0.1 width (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
    new_bin_edges = np.arange(0.0, 1.1, 0.1)
    new_bin_centers = new_bin_edges[:-1] + 0.05  # Centers at 0.05, 0.15, ..., 0.95
    
    # Combine counts into new bins
    new_draft_counts = np.zeros(len(new_bin_centers))
    new_reject_counts = np.zeros(len(new_bin_centers))
    
    for i, (draft_count, reject_count, bin_start) in enumerate(zip(draft_counts, reject_counts, bin_starts)):
        # Find which new bin this old bin belongs to
        bin_idx = int(bin_start * 10)  # 0.00->0, 0.02->0, ..., 0.98->9
        if bin_idx < len(new_bin_centers):
            new_draft_counts[bin_idx] += draft_count
            new_reject_counts[bin_idx] += reject_count
    
    # Use the new combined bins
    bin_centers = new_bin_centers
    draft_counts = new_draft_counts
    reject_counts = new_reject_counts
    
    # Calculate percentages
    draft_total = np.sum(draft_counts)
    reject_total = np.sum(reject_counts)
    
    # Calculate accepted tokens = draft tokens - rejected tokens for each bin
    accepted_counts = draft_counts - reject_counts
    accepted_total = np.sum(accepted_counts)
    
    total_tokens = draft_total + reject_total
    
    accepted_percentages = (accepted_counts / total_tokens) * 100
    reject_percentages = (reject_counts / total_tokens) * 100
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate bin width (now using 0.1 width bins)
    bin_width = 0.08
    
    # Create stacked bar plot
    ax.bar(bin_centers, accepted_percentages, width=bin_width*0.8, 
           alpha=1.0, color="#2F5597", edgecolor='black', linewidth=0.5, 
           label='Accepted', zorder=3)
    ax.bar(bin_centers, reject_percentages, width=bin_width*0.8, 
           bottom=accepted_percentages, alpha=1.0, color='#FFC000', 
           edgecolor='black', linewidth=0.5, label='Rejected', zorder=3)
    
    # Set labels and formatting
    ax.set_xlabel('Confidence', fontsize=24)
    ax.set_ylabel('Token Distribution (%)', fontsize=24)
    # ax.set_title('Accepted vs Rejected Tokens Distribution (Stacked)', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=24)
    
    # Set x-axis properties
    # Create custom tick positions for 0.2 width intervals
    # Ticks at 0.2, 0.4, 0.6, 0.8 positioned between pairs of bins
    custom_tick_positions = [0.2, 0.4, 0.6, 0.8]
    custom_tick_labels = ['0.2', '0.4', '0.6', '0.8']
    
    ax.set_xticks(custom_tick_positions)
    ax.set_xticklabels(custom_tick_labels)
    ax.set_xlim(-0.05, 1.05)
    
    # Set y-axis limit and increase tick fontsize
    ax.set_ylim(0, 35)
    ax.tick_params(axis='both', labelsize=24)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Stacked figure saved to: {output_path}")
    
    # Print additional statistics for accepted tokens
    print(f"\nStacked Histogram Statistics:")
    print(f"Total draft tokens: {draft_total}")
    print(f"Total reject tokens: {reject_total}")
    print(f"Total accepted tokens: {accepted_total}")
    print(f"Acceptance rate: {(accepted_total / draft_total) * 100:.2f}%")
    
    return fig, ax

def create_stacked_histogram_original_bins(csv_path, output_path=None, figsize=(8, 6)):
    """
    Create a stacked histogram plot with original bin widths (no combining).
    
    Args:
        csv_path: Path to the CSV file containing histogram data
        output_path: Path to save the output figure (optional)
        figsize: Figure size tuple
    """
    # Read the CSV data
    df = pd.read_csv(csv_path)
    
    # Find draft and reject rows
    draft_rows = df[df['experiment'].str.contains('draft')]
    reject_rows = df[df['experiment'].str.contains('reject') & ~df['experiment'].str.contains('budget')]
    
    if draft_rows.empty or reject_rows.empty:
        raise ValueError("Could not find both draft and reject data in the CSV file")
    
    # Use the first draft and reject rows
    draft_row = draft_rows.iloc[0]
    reject_row = reject_rows.iloc[0]
    
    # Get the bin columns (exclude 'experiment' column)
    bin_columns = [col for col in df.columns if col != 'experiment']
    
    # Extract counts for draft and reject
    draft_counts = draft_row[bin_columns].values
    reject_counts = reject_row[bin_columns].values
    
    # Convert bin labels to center values
    bin_centers = []
    bin_widths = []
    for bin_label in bin_columns:
        # Extract start and end values from "0.00-0.02" format
        start, end = map(float, bin_label.split('-'))
        center = (start + end) / 2
        width = end - start
        bin_centers.append(center)
        bin_widths.append(width)
    
    bin_centers = np.array(bin_centers)
    bin_widths = np.array(bin_widths)
    
    # Calculate percentages
    draft_total = np.sum(draft_counts)
    reject_total = np.sum(reject_counts)
    
    # Calculate accepted tokens = draft tokens - rejected tokens for each bin
    accepted_counts = draft_counts - reject_counts
    accepted_total = np.sum(accepted_counts)
    
    total_tokens = draft_total + reject_total
    
    accepted_percentages = (accepted_counts / total_tokens) * 100
    reject_percentages = (reject_counts / total_tokens) * 100
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use 80% of the bin width for the bars
    bar_widths = bin_widths * 1.0
    
    # Create stacked bar plot
    ax.bar(bin_centers, accepted_percentages, width=bar_widths, 
           alpha=1.0, color="#2F5597", edgecolor='black', linewidth=0.5, 
           label='Accepted', zorder=3)
    ax.bar(bin_centers, reject_percentages, width=bar_widths, 
           bottom=accepted_percentages, alpha=1.0, color='#FFC000', 
           edgecolor='black', linewidth=0.5, label='Rejected', zorder=3)
    
    # Set labels and formatting
    ax.set_xlabel('Confidence', fontsize=24)
    ax.set_ylabel('Token Distribution (%)', fontsize=24)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=24)
    
    # Set x-axis properties
    ax.set_xlim(-0.05, 1.05)
    
    # Set y-axis limit and increase tick fontsize
    ax.set_ylim(0, 25)
    ax.tick_params(axis='both', labelsize=24)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Stacked figure (original bins) saved to: {output_path}")
    
    # Print statistics
    print(f"\nStacked Histogram (Original Bins) Statistics:")
    print(f"Total draft tokens: {draft_total}")
    print(f"Total reject tokens: {reject_total}")
    print(f"Total accepted tokens: {accepted_total}")
    print(f"Acceptance rate: {(accepted_total / draft_total) * 100:.2f}%")
    print(f"Number of bins: {len(bin_centers)}")
    print(f"Bin width: {bin_widths[0]:.4f}")
    
    return fig, ax

def main():
    """Main function to create histogram plots."""
    # Configuration
    confidence_dir = "/home/juchanlee/MagicDec/figure/confidence"
    csv_file = "run2step_Meta-Llama-3.1-8B_pg19_histogram_data_32800_new.csv"
    csv_path = os.path.join(confidence_dir, csv_file)
    
    # Output paths
    output_dir = "/home/juchanlee/MagicDec/figure/confidence"
    dual_axis_output = os.path.join(output_dir, "combined_histogram_dual_axis.png")
    stacked_output = os.path.join(output_dir, "combined_histogram_stacked.png")
    stacked_original_output = os.path.join(output_dir, "combined_histogram_stacked_original_bins.png")
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    print(f"Reading data from: {csv_path}")
    
    try:
        # Create dual-axis histogram
        print("\nCreating dual-axis histogram...")
        fig1, ax1, ax2 = create_combined_histogram(csv_path, dual_axis_output)
        
        # Create stacked histogram (combined bins)
        print("\nCreating stacked histogram (combined bins)...")
        fig2, ax3 = create_stacked_histogram(csv_path, stacked_output)
        
        # Create stacked histogram (original bins)
        print("\nCreating stacked histogram (original bins)...")
        fig3, ax4 = create_stacked_histogram_original_bins(csv_path, stacked_original_output)
        
        # Show the plots
        plt.show()
        
    except Exception as e:
        print(f"Error creating histograms: {e}")

if __name__ == "__main__":
    main()