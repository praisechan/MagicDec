#!/usr/bin/env python3
"""
Create histogram visualization for run2step confidence data
Generates a two-panel plot: normal tokens (draft) and rejected tokens
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV data
csv_path = "/home/juchanlee/MagicDec/figure/confidence/run2step_Meta-Llama-3.1-8B_pg19_histogram_data_16416_new.csv"
df = pd.read_csv(csv_path)

# Extract draft and reject data
draft_row = df[df['experiment'].str.contains('draft')].iloc[0]
reject_row = df[df['experiment'].str.contains('reject')].iloc[0]

# Get the bin columns (exclude 'experiment' column)
bin_columns = [col for col in df.columns if col != 'experiment']

# Extract counts for draft and reject
draft_counts = draft_row[bin_columns].values
reject_counts = reject_row[bin_columns].values

# Convert bin labels to center values for x-axis
bin_centers = []
for bin_label in bin_columns:
    # Extract start and end values from "0.00-0.02" format
    start, end = map(float, bin_label.split('-'))
    center = (start + end) / 2
    bin_centers.append(center)

bin_centers = np.array(bin_centers)

# Calculate percentages
draft_total = np.sum(draft_counts)
reject_total = np.sum(reject_counts)

draft_percentages = (draft_counts / draft_total) * 100
reject_percentages = (reject_counts / reject_total) * 100

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Calculate bin width for bar plots
bin_width = bin_centers[1] - bin_centers[0]

# Upper plot: Draft tokens (normal tokens)
ax1.bar(bin_centers, draft_percentages, width=bin_width*0.8, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Percentage (%)', fontsize=12)
ax1.set_title('Draft Tokens (Normal)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, max(draft_percentages) * 1.1)

# Lower plot: Reject tokens
ax2.bar(bin_centers, reject_percentages, width=bin_width*0.8, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_title('Rejected Tokens', fontsize=14, fontweight='bold')
ax2.set_xlabel('Top1-Top2 Difference', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, max(reject_percentages) * 1.1)

# Set x-axis ticks to show 0.1, 0.2, ..., 0.9
x_ticks = np.arange(0.1, 1.0, 0.1)
ax2.set_xticks(x_ticks)
ax2.set_xticklabels([f'{x:.1f}' for x in x_ticks])

# Set x-axis limits
ax1.set_xlim(0, 1)
ax2.set_xlim(0, 1)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
output_path = "/home/juchanlee/MagicDec/figure/confidence/run2step_histogram_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"Figure saved to: {output_path}")

# Print some statistics
print(f"\nStatistics:")
print(f"Total draft tokens: {draft_total}")
print(f"Total reject tokens: {reject_total}")
print(f"Draft acceptance rate: {(draft_total / (draft_total + reject_total)) * 100:.2f}%")
print(f"Rejection rate: {(reject_total / (draft_total + reject_total)) * 100:.2f}%")

# Show the plot
plt.show()