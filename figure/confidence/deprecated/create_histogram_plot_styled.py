#!/usr/bin/env python3
"""
Create histogram visualization for run2step confidence data
Generates a two-panel plot similar to the reference image style
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set font to serif (Times-like appearance)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Read the CSV data
prefix_len = 32800
csv_path = f"/home/juchanlee/MagicDec/figure/confidence/run2step_Meta-Llama-3.1-8B_pg19_histogram_data_{prefix_len}.csv"
df = pd.read_csv(csv_path)

# Extract draft and reject data
draft_row = df[df['experiment'].str.contains('draft')].iloc[0]
reject_row = df[df['experiment'].str.contains('reject')].iloc[0]

# Get the bin columns (exclude 'experiment' column)
bin_columns = [col for col in df.columns if col != 'experiment']

# Extract counts for draft and reject
draft_counts = draft_row[bin_columns].values
reject_counts = reject_row[bin_columns].values

# Combine every two bins into one to reduce the number of bins
combined_draft_counts = []
combined_reject_counts = []
combined_bin_labels = []
combined_bin_centers = []

for i in range(0, len(bin_columns), 2):
    # Combine two consecutive bins
    if i + 1 < len(bin_columns):
        # Sum counts from two bins
        combined_draft = draft_counts[i] + draft_counts[i + 1]
        combined_reject = reject_counts[i] + reject_counts[i + 1]
        
        # Create new bin label spanning both original bins
        start_bin = bin_columns[i]
        end_bin = bin_columns[i + 1]
        start_val = float(start_bin.split('-')[0])
        end_val = float(end_bin.split('-')[1])
        combined_label = f"{start_val:.2f}-{end_val:.2f}"
        
        # Calculate center of combined bin
        combined_center = (start_val + end_val) / 2
    else:
        # Handle odd number of bins - keep the last bin as is
        combined_draft = draft_counts[i]
        combined_reject = reject_counts[i]
        combined_label = bin_columns[i]
        start_val, end_val = map(float, bin_columns[i].split('-'))
        combined_center = (start_val + end_val) / 2
    
    combined_draft_counts.append(combined_draft)
    combined_reject_counts.append(combined_reject)
    combined_bin_labels.append(combined_label)
    combined_bin_centers.append(combined_center)

# Convert to numpy arrays
combined_draft_counts = np.array(combined_draft_counts)
combined_reject_counts = np.array(combined_reject_counts)
combined_bin_centers = np.array(combined_bin_centers)

# Calculate percentages (density-like)
draft_total = np.sum(combined_draft_counts)
reject_total = np.sum(combined_reject_counts)

draft_percentages = (combined_draft_counts / draft_total) * 100
reject_percentages = (combined_reject_counts / reject_total) * 100

# Create the figure with two subplots - make them closer together
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Calculate bin width for bar plots
bin_width = combined_bin_centers[1] - combined_bin_centers[0]

# Color scheme similar to the reference
bar_color = '#4682B4'  # Steel blue
edge_color = '#2F4F4F'  # Dark slate gray

# Calculate the maximum y-value to use the same scale for both plots
max_y_value = max(max(draft_percentages), max(reject_percentages)) * 1.05

# Upper plot: Draft tokens (normal tokens)
bars1 = ax1.bar(combined_bin_centers, draft_percentages, width=bin_width*0.9, 
                alpha=0.8, color=bar_color, edgecolor=edge_color, linewidth=0.5)
# Move title inside the plot area, slightly down from top
ax1.text(0.05, 0.85, 'Draft Tokens', transform=ax1.transAxes, fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_ylim(0, max_y_value)

# Lower plot: Reject tokens
bars2 = ax2.bar(combined_bin_centers, reject_percentages, width=bin_width*0.9, 
                alpha=0.8, color=bar_color, edgecolor=edge_color, linewidth=0.5)
# Move title inside the plot area, slightly down from top
ax2.text(0.05, 0.85, 'Rejected Tokens', transform=ax2.transAxes, fontsize=16, fontweight='bold')
# Remove x-axis label - ax2.set_xlabel('Top1-Top2 Difference', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_ylim(0, max_y_value)

# Set x-axis ticks to show 0.1, 0.2, ..., 0.9
x_ticks = np.arange(0.1, 1.0, 0.1)
ax2.set_xticks(x_ticks)
ax2.set_xticklabels([f'{x:.1f}' for x in x_ticks], fontsize=12)

# Set x-axis limits
ax1.set_xlim(0, 1)
ax2.set_xlim(0, 1)

# Remove shading - no background highlighting
# high_conf_start = 0.8
# ax1.axvspan(high_conf_start, 1.0, alpha=0.2, color='gray', zorder=0)
# low_conf_end = 0.2
# ax2.axvspan(0.0, low_conf_end, alpha=0.2, color='gray', zorder=0)

# Style improvements
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

# Remove y-axis label - no longer adding it to the figure
# fig.text(0.04, 0.5, 'Percentage (%)', va='center', rotation='vertical', 
#          fontsize=14, fontweight='bold')

# Remove individual y-axis labels since we have a common one
ax1.set_ylabel('')
ax2.set_ylabel('')

# Adjust layout to make graphs concatenated with minimal space
plt.tight_layout()
plt.subplots_adjust(hspace=0.05)  # Very small space between subplots, no extra left margin needed

# Save the figure with titles
output_path_with_titles = f"/home/juchanlee/MagicDec/figure/confidence/run2step_histogram_styled_{prefix_len}.png"
plt.savefig(output_path_with_titles, dpi=300, bbox_inches='tight', facecolor='white')

print(f"Styled figure with titles saved to: {output_path_with_titles}")

# Create version without titles
# Remove all text objects from both axes
for txt in ax1.texts:
    txt.remove()
for txt in ax2.texts:
    txt.remove()

# Save the figure without titles
output_path_no_titles = f"/home/juchanlee/MagicDec/figure/confidence/run2step_histogram_styled_{prefix_len}_no_titles.png"
plt.savefig(output_path_no_titles, dpi=300, bbox_inches='tight', facecolor='white')

print(f"Styled figure without titles saved to: {output_path_no_titles}")

# Print some statistics
print(f"\nStatistics:")
print(f"Total draft tokens: {draft_total}")
print(f"Total reject tokens: {reject_total}")
print(f"Draft acceptance rate: {(draft_total / (draft_total + reject_total)) * 100:.2f}%")
print(f"Rejection rate: {(reject_total / (draft_total + reject_total)) * 100:.2f}%")

# Print top confidence ranges for draft tokens
print(f"\nTop confidence ranges for draft tokens:")
top_indices = np.argsort(draft_percentages)[-5:][::-1]
for idx in top_indices:
    print(f"  {combined_bin_labels[idx]}: {draft_percentages[idx]:.2f}% ({combined_draft_counts[idx]} tokens)")

print(f"\nTop confidence ranges for reject tokens:")
top_indices = np.argsort(reject_percentages)[-5:][::-1]
for idx in top_indices:
    if reject_percentages[idx] > 0:
        print(f"  {combined_bin_labels[idx]}: {reject_percentages[idx]:.2f}% ({combined_reject_counts[idx]} tokens)")

# Show the plot
plt.show()