#!/usr/bin/env python3
"""
Create histogram visualization for run2step confidence data with multiple budget types
Generates a two-panel plot: one for draft tokens and one for reject tokens with different budgets
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
csv_path = f"/home/juchanlee/MagicDec/figure/confidence/run2step_Meta-Llama-3.1-8B_pg19_histogram_data_{prefix_len}_new.csv"
df = pd.read_csv(csv_path)

# Extract draft data
draft_row = df[df['experiment'].str.contains('draft')].iloc[0]

# Extract reject data for different budgets in desired order (budget types first, then baseline)
reject_rows = {}

# Extract budget-specific reject data first (these will appear first in the bars)
budget_values = ['0.10', '0.25', '0.40']
for budget in budget_values:
    budget_rows = df[df['experiment'].str.contains(f'budget_{budget}_reject')]
    if not budget_rows.empty:
        reject_rows[f'budget_{budget}'] = budget_rows.iloc[0]

# Find baseline reject row last (this will appear last in the bars)
baseline_rows = df[df['experiment'].str.contains('report_reject$') & ~df['experiment'].str.contains('budget_[0-9]')]
if not baseline_rows.empty:
    reject_rows['baseline'] = baseline_rows.iloc[0]

# Get the bin columns (exclude 'experiment' column)
bin_columns = [col for col in df.columns if col != 'experiment']

# Extract counts for draft
draft_counts = draft_row[bin_columns].values

# Extract counts for all reject types
reject_counts = {}
for key, row in reject_rows.items():
    reject_counts[key] = row[bin_columns].values

# Group bins into 0.1 width intervals (combine 5 bins of 0.02 width each)
combined_draft_counts = []
combined_reject_counts = {key: [] for key in reject_counts.keys()}
combined_bin_labels = []
combined_bin_centers = []

# Each 0.1 interval should contain 5 original bins (since each original bin is 0.02 width)
bins_per_group = 5

for i in range(0, len(bin_columns), bins_per_group):
    # Determine the range for this group
    end_idx = min(i + bins_per_group, len(bin_columns))
    
    # Sum counts from all bins in this group
    combined_draft = np.sum(draft_counts[i:end_idx])
    
    # Create new bin label for 0.1 width interval
    start_bin = bin_columns[i]
    end_bin = bin_columns[end_idx - 1]
    start_val = float(start_bin.split('-')[0])
    end_val = float(end_bin.split('-')[1])
    combined_label = f"{start_val:.1f}-{end_val:.1f}"
    
    # Calculate center of combined bin
    combined_center = (start_val + end_val) / 2
    
    # Combine reject counts for all budget types
    for key in reject_counts.keys():
        combined_reject = np.sum(reject_counts[key][i:end_idx])
        combined_reject_counts[key].append(combined_reject)
    
    combined_draft_counts.append(combined_draft)
    combined_bin_labels.append(combined_label)
    combined_bin_centers.append(combined_center)

# Convert to numpy arrays
combined_draft_counts = np.array(combined_draft_counts)
combined_bin_centers = np.array(combined_bin_centers)
for key in combined_reject_counts.keys():
    combined_reject_counts[key] = np.array(combined_reject_counts[key])

# Use frequencies instead of percentages for draft tokens
draft_total = np.sum(combined_draft_counts)
draft_frequencies = combined_draft_counts  # Use raw counts as frequencies

# Calculate frequencies for reject data - use raw counts
reject_frequencies = {}
reject_totals = {}

# First, calculate totals for all reject types
for key in combined_reject_counts.keys():
    reject_totals[key] = np.sum(combined_reject_counts[key])

# Calculate frequencies - use raw counts for all budget types
for key in combined_reject_counts.keys():
    if reject_totals[key] > 0:
        # All budgets: use raw counts as frequencies
        reject_frequencies[key] = combined_reject_counts[key]
    else:
        reject_frequencies[key] = np.zeros_like(combined_reject_counts[key])

# Create the figure with two subplots (horizontal layout)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Remove sharey=True for separate y-axis scales

# Calculate bin width for bar plots
bin_width = combined_bin_centers[1] - combined_bin_centers[0]

# Color scheme for draft tokens
draft_color = '#4682B4'  # Steel blue
edge_color = '#2F4F4F'  # Dark slate gray

# Color scheme for different budget types
budget_colors = {
    'baseline': '#FF6B6B',      # Red
    'budget_0.10': '#4ECDC4',   # Teal
    'budget_0.25': '#45B7D1',   # Blue
    'budget_0.40': '#96CEB4'    # Green
}

# Calculate the maximum y-value for both plots (separate scales for fine-grained reject plot)
max_draft_y = max(draft_frequencies) * 1.05
max_reject_y = 0
if reject_frequencies:
    max_reject_y = max([max(frequencies) for frequencies in reject_frequencies.values()]) * 1.05

# Left plot: Draft tokens
bars1 = ax1.bar(combined_bin_centers, draft_frequencies, width=bin_width*0.9, 
                alpha=0.8, color=draft_color, edgecolor=edge_color, linewidth=0.5)
ax1.text(0.05, 0.85, 'Draft Tokens', transform=ax1.transAxes, fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_ylim(0, max_draft_y)

# Right plot: Reject tokens with multiple budgets (grouped bars)
num_budgets = len(reject_frequencies)
if num_budgets > 0:
    # Calculate individual bar width and positions
    individual_bar_width = bin_width * 0.8 / num_budgets
    
    # Create grouped bars
    budget_labels = []
    legend_handles = []
    
    for i, (budget_key, frequencies) in enumerate(reject_frequencies.items()):
        # Calculate x positions for this budget type
        x_positions = combined_bin_centers - (bin_width * 0.4) + (i + 0.5) * individual_bar_width
        
        # Create legend label based on budget type
        if budget_key == 'baseline':
            legend_label = '100% KV'
        else:
            # Extract budget value and convert to percentage
            budget_value = budget_key.replace('budget_', '')
            budget_percentage = int(float(budget_value) * 100)
            legend_label = f'{budget_percentage}% KV'
        
        # Create bars for this budget
        color = budget_colors.get(budget_key, '#888888')
        bars = ax2.bar(x_positions, frequencies, width=individual_bar_width, 
                      alpha=0.8, color=color, edgecolor=edge_color, linewidth=0.3,
                      label=legend_label)
        legend_handles.append(bars[0])
        
        # Store label for legend
        budget_labels.append(legend_label)

    # Add legend to the reject tokens plot
    ax2.legend(legend_handles, budget_labels, loc='upper right', fontsize=10, framealpha=0.9)

ax2.text(0.05, 0.85, 'Rejected Tokens', transform=ax2.transAxes, fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_ylim(0, max_reject_y)  # Use separate y-axis scale for reject plot

# Set x-axis ticks to show 0.1, 0.2, ..., 0.9 for both plots
x_ticks = np.arange(0.1, 1.0, 0.1)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels([f'{x:.1f}' for x in x_ticks], fontsize=12)
ax2.set_xticks(x_ticks)
ax2.set_xticklabels([f'{x:.1f}' for x in x_ticks], fontsize=12)

# Set x-axis limits for both plots
ax1.set_xlim(0, 1)
ax2.set_xlim(0, 1)

# Style improvements
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

# Add x-axis labels for both plots
ax1.set_xlabel('Confidence Score', fontsize=14)
ax2.set_xlabel('Confidence Score', fontsize=14)

# Set y-axis labels for both plots (now using separate scales)
ax1.set_ylabel('Frequency (Draft Tokens)', fontsize=14)
ax2.set_ylabel('Frequency (Reject Tokens)', fontsize=14)

# Adjust layout to make graphs side by side with minimal space
plt.tight_layout()
plt.subplots_adjust(wspace=0.15)  # Small space between plots horizontally

# Save the figure with titles
output_path_with_titles = f"/home/juchanlee/MagicDec/figure/confidence/run2step_histogram_multibudget_styled_{prefix_len}.png"
plt.savefig(output_path_with_titles, dpi=300, bbox_inches='tight', facecolor='white')

print(f"Multi-budget styled figure with titles saved to: {output_path_with_titles}")

# Create version without titles
# Remove all text objects from both axes
for txt in ax1.texts:
    txt.remove()
for txt in ax2.texts:
    txt.remove()

# Save the figure without titles
output_path_no_titles = f"/home/juchanlee/MagicDec/figure/confidence/run2step_histogram_multibudget_styled_{prefix_len}_no_titles.png"
plt.savefig(output_path_no_titles, dpi=300, bbox_inches='tight', facecolor='white')

print(f"Multi-budget styled figure without titles saved to: {output_path_no_titles}")

# Print some statistics
print(f"\nStatistics:")
print(f"Total draft tokens: {draft_total}")

total_reject_tokens = sum(reject_totals.values())
print(f"Total reject tokens (all budgets): {total_reject_tokens}")
print(f"Draft acceptance rate: {(draft_total / (draft_total + total_reject_tokens)) * 100:.2f}%")

print(f"\nReject tokens by budget:")
for budget_key, total in reject_totals.items():
    if budget_key == 'baseline':
        label = '100% KV'
    else:
        budget_value = budget_key.replace('budget_', '')
        budget_percentage = int(float(budget_value) * 100)
        label = f'{budget_percentage}% KV'
    print(f"  {label}: {total} tokens ({(total/total_reject_tokens)*100:.2f}% of rejects)")

# Print top confidence ranges for draft tokens
print(f"\nTop confidence ranges for draft tokens:")
top_indices = np.argsort(draft_frequencies)[-5:][::-1]
for idx in top_indices:
    percentage = (draft_frequencies[idx] / draft_total) * 100
    print(f"  {combined_bin_labels[idx]}: {draft_frequencies[idx]} tokens ({percentage:.2f}%)")

# Print top confidence ranges for each reject budget
for budget_key, frequencies in reject_frequencies.items():
    if reject_totals[budget_key] > 0:
        if budget_key == 'baseline':
            label = '100% KV'
        else:
            budget_value = budget_key.replace('budget_', '')
            budget_percentage = int(float(budget_value) * 100)
            label = f'{budget_percentage}% KV'
        print(f"\nTop confidence ranges for {label} reject tokens (frequencies):")
        top_indices = np.argsort(frequencies)[-3:][::-1]  # Show top 3 for each
        for idx in top_indices:
            if frequencies[idx] > 0:
                percentage = (combined_reject_counts[budget_key][idx] / reject_totals[budget_key]) * 100
                print(f"  {combined_bin_labels[idx]}: {frequencies[idx]:.1f} frequency ({combined_reject_counts[budget_key][idx]} tokens, {percentage:.2f}%)")

# Show the plot
plt.show()