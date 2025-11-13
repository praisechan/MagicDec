import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Read the CSV file
csv_file = '/home/juchanlee/MagicDec/figure/confidence/data/run2step_Meta-Llama-3.1-8B_pg19_histogram_data_32768_multibudget.csv'
df = pd.read_csv(csv_file)

# Extract the relevant rows
draft_row = df[df['experiment'].str.contains('report_draft')]
# The baseline reject is budget0.1 (note: no underscore before 0.1)
reject_row = df[df['experiment'].str.contains('budget0.1_taskgov_report_reject') & ~df['experiment'].str.contains('budget_0')]
budget_020_row = df[df['experiment'].str.contains('budget_0.20_reject')]
budget_040_row = df[df['experiment'].str.contains('budget_0.40_reject')]

# Get the bin columns (exclude 'experiment' column)
bin_columns = [col for col in df.columns if col != 'experiment']

# Extract values
draft_values = draft_row[bin_columns].values[0]
reject_values = reject_row[bin_columns].values[0]
budget_020_values = budget_020_row[bin_columns].values[0]
budget_040_values = budget_040_row[bin_columns].values[0]

# Calculate accepted tokens (draft - reject)
accepted_values = draft_values - reject_values

# Combine bins (0.02 * 5 = 0.1 width)
n_bins_to_combine = 5
n_combined_bins = len(bin_columns) // n_bins_to_combine

combined_draft = []
combined_accepted = []
combined_reject = []
combined_budget_020 = []
combined_budget_040 = []
combined_bin_centers = []

for i in range(n_combined_bins):
    start_idx = i * n_bins_to_combine
    end_idx = start_idx + n_bins_to_combine
    
    # Sum the values in each group of 5 bins
    combined_draft.append(np.sum(draft_values[start_idx:end_idx]))
    combined_accepted.append(np.sum(accepted_values[start_idx:end_idx]))
    combined_reject.append(np.sum(reject_values[start_idx:end_idx]))
    combined_budget_020.append(np.sum(budget_020_values[start_idx:end_idx]))
    combined_budget_040.append(np.sum(budget_040_values[start_idx:end_idx]))
    
    # Bin center is at the middle of the 0.1 range
    combined_bin_centers.append(i * 0.1 + 0.05)

combined_draft = np.array(combined_draft)
combined_accepted = np.array(combined_accepted)
combined_reject = np.array(combined_reject)
combined_budget_020 = np.array(combined_budget_020)
combined_budget_040 = np.array(combined_budget_040)
combined_bin_centers = np.array(combined_bin_centers)

# Create the plot with ISCA-standard figure size (column width ~3.5", full width ~7")
fig, ax1 = plt.subplots(figsize=(7, 3.5))

# ISCA-style colors (colorblind-friendly)
color_accepted = '#377eb8'  # Blue
color_rejected = '#ff7f00'  # Orange
color_actual = '#e41a1c'    # Red
color_budget20 = '#4daf4a'  # Green
color_budget40 = '#984ea3'  # Purple

# Bar chart: draft tokens only
bar_width = 0.08
bars_draft = ax1.bar(combined_bin_centers, combined_draft, width=bar_width, 
                     alpha=0.9, color=color_accepted, edgecolor='black', linewidth=1.2,
                     label='Draft Tokens')

# Configure left y-axis (histogram)
ax1.set_xlabel('Token Probability', fontsize=14, fontweight='bold')
ax1.set_ylabel('# Tokens', fontsize=14, fontweight='bold')
ax1.tick_params(axis='both', labelsize=12, width=1.5)
ax1.set_xticks(np.arange(0, 1.2, 0.2))
ax1.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
ax1.set_axisbelow(True)

# Create second y-axis for reject counts
ax2 = ax1.twinx()

# Line plots for different budgets with distinct styles
line_actual = ax2.plot(combined_bin_centers, combined_reject, 
                       color=color_actual, marker='o', linewidth=2.5, markersize=7,
                       label='Baseline', alpha=1.0,
                       markeredgecolor='white', markeredgewidth=1.0, zorder=5)
line_020 = ax2.plot(combined_bin_centers, combined_budget_020, 
                    color=color_budget20, marker='s', linewidth=2.5, markersize=6,
                    label='Budget 0.20', alpha=1.0, linestyle='--',
                    markeredgecolor='white', markeredgewidth=1.0, zorder=4)
line_040 = ax2.plot(combined_bin_centers, combined_budget_040, 
                    color=color_budget40, marker='^', linewidth=2.5, markersize=7,
                    label='Budget 0.40', alpha=1.0, linestyle='-.',
                    markeredgecolor='white', markeredgewidth=1.0, zorder=4)

ax2.set_ylabel('# Rejected Tokens', fontsize=14, fontweight='bold')
ax2.tick_params(axis='y', labelsize=12, width=1.5)
max_reject = max(np.max(combined_reject), np.max(combined_budget_020), 
                 np.max(combined_budget_040))
ax2.set_ylim(0, max_reject * 1.15)
ax2.spines['right'].set_linewidth(1.5)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['top'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)

# Combine legends - ISCA style with frame
bars_handles, bars_labels = ax1.get_legend_handles_labels()
lines_handles = line_actual + line_020 + line_040
lines_labels = ['Baseline', 'Budget 0.20', 'Budget 0.40']

# Place legend at upper center with ISCA-style formatting
ax1.legend(bars_handles + lines_handles, bars_labels + lines_labels,
           loc='upper center', bbox_to_anchor=(0.5, 1.02),
           frameon=True, fontsize=10, ncol=4,
           columnspacing=1.0, handlelength=2.0, handletextpad=0.5,
           fancybox=False, edgecolor='black', framealpha=0.95)

plt.tight_layout()
# Save as both PNG and PDF (PDF is preferred for publications)
plt.savefig('/home/juchanlee/MagicDec/figure/confidence/histogram_combined_with_budgets.png', 
            dpi=300, bbox_inches='tight')
plt.savefig('/home/juchanlee/MagicDec/figure/confidence/histogram_combined_with_budgets.pdf', 
            bbox_inches='tight')
print("Saved: histogram_combined_with_budgets.png")
print("Saved: histogram_combined_with_budgets.pdf")

plt.show()

# Print statistics
print("\n=== Statistics ===")
print(f"Total draft tokens: {np.sum(combined_draft):.0f}")
print(f"Total accepted tokens: {np.sum(combined_accepted):.0f}")
print(f"Total rejected tokens (actual): {np.sum(combined_reject):.0f}")
print(f"Overall reject rate: {np.sum(combined_reject) / np.sum(combined_draft) * 100:.2f}%")
print(f"\nTotal rejected tokens (budget 0.20): {np.sum(combined_budget_020):.0f}")
print(f"Total rejected tokens (budget 0.40): {np.sum(combined_budget_040):.0f}")
print(f"\nVerification: Draft = Accepted + Rejected?")
print(f"  {np.sum(combined_draft):.0f} = {np.sum(combined_accepted):.0f} + {np.sum(combined_reject):.0f}")
print(f"  Match: {np.isclose(np.sum(combined_draft), np.sum(combined_accepted) + np.sum(combined_reject))}")
