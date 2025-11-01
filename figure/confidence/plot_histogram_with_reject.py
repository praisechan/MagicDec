import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
csv_file = '/home/juchanlee/MagicDec/figure/confidence/data/run2step_Meta-Llama-3.1-8B_pg19_histogram_data_32768.csv'
df = pd.read_csv(csv_file)

# Extract the relevant rows
draft_row = df[df['experiment'] == 'run2step_Meta-Llama-3.1-8B_pg19_prefix32768_gamma1_4_budget0.1_taskgov_report_draft_top1_logits']
reject_row = df[df['experiment'] == 'run2step_Meta-Llama-3.1-8B_pg19_prefix32768_gamma1_4_budget0.1_taskgov_report_reject_top1_logits']

# Get the bin columns (exclude 'experiment' column)
bin_columns = [col for col in df.columns if col != 'experiment']

# Extract values
draft_values = draft_row[bin_columns].values[0]
reject_values = reject_row[bin_columns].values[0]

# Calculate reject percentage for each bin
reject_percentage = np.divide(reject_values, draft_values, 
                              out=np.zeros_like(reject_values, dtype=float), 
                              where=draft_values!=0) * 100

# Create bin centers for x-axis
bin_centers = np.arange(0.01, 1.0, 0.02)

# # ========== VERSION 1: Raw data (0.02 width bins) ==========
# fig1, ax1 = plt.subplots(figsize=(12, 6))

# # Bar graph for draft distribution
# bars = ax1.bar(bin_centers, draft_values, width=0.018, alpha=0.6, 
#                color='steelblue', label='Draft Tokens Distribution')

# # Set x-axis ticks at 0.2 intervals
# ax1.set_xticks(np.arange(0, 1.2, 0.2))
# ax1.set_xlabel('Confidence (Top-1 Logit)', fontsize=12)
# ax1.set_ylabel('Number of Tokens', fontsize=12)
# ax1.set_title('Draft Token Distribution and Reject Percentage (0.02 bin width)', fontsize=14)
# ax1.grid(True, alpha=0.3, axis='y')

# # Create second y-axis for reject percentage
# ax2 = ax1.twinx()
# line = ax2.plot(bin_centers, reject_percentage, 'r-o', linewidth=2, 
#                 markersize=4, label='Reject Percentage', alpha=0.8)
# ax2.set_ylabel('Reject Percentage (%)', fontsize=12, color='red')
# ax2.tick_params(axis='y', labelcolor='red')
# ax2.set_ylim(0, max(reject_percentage) * 1.1 if max(reject_percentage) > 0 else 100)

# # Combine legends
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# plt.tight_layout()
# plt.savefig('/home/juchanlee/MagicDec/figure/confidence/histogram_reject_raw.png', dpi=300, bbox_inches='tight')
# print("Saved: histogram_reject_raw.png")

# ========== VERSION 2: Combined bins (0.1 width) ==========
# Combine every 5 bins (0.02 * 5 = 0.1)
n_bins_to_combine = 5
n_combined_bins = len(bin_columns) // n_bins_to_combine

combined_draft = []
combined_reject = []
combined_bin_centers = []

for i in range(n_combined_bins):
    start_idx = i * n_bins_to_combine
    end_idx = start_idx + n_bins_to_combine
    
    # Sum the values in each group of 5 bins
    combined_draft.append(np.sum(draft_values[start_idx:end_idx]))
    combined_reject.append(np.sum(reject_values[start_idx:end_idx]))
    
    # Bin center is at the middle of the 0.1 range
    combined_bin_centers.append(i * 0.1 + 0.05)

combined_draft = np.array(combined_draft)
combined_reject = np.array(combined_reject)
combined_bin_centers = np.array(combined_bin_centers)

# Calculate reject percentage for combined bins
combined_reject_percentage = np.divide(combined_reject, combined_draft,
                                       out=np.zeros_like(combined_reject, dtype=float),
                                       where=combined_draft!=0) * 100

fig2, ax3 = plt.subplots(figsize=(8, 5))

# Bar graph for draft distribution (combined bins)
bars2 = ax3.bar(combined_bin_centers, combined_draft, width=0.08, alpha=1,
                color='#2F5597', edgecolor='black', label='# Draft Tokens')

# Set x-axis ticks at 0.2 intervals
ax3.set_xticks(np.arange(0, 1.2, 0.2))
ax3.set_xlabel('Confidence (Top-1 Logit)', fontsize=16)
ax3.set_ylabel('# Draft Tokens', fontsize=16)
ax3.tick_params(axis='both', labelsize=16)
ax3.grid(True, alpha=0.3, axis='y')

# Create second y-axis for reject percentage
ax4 = ax3.twinx()
line2 = ax4.plot(combined_bin_centers, combined_reject_percentage, 'r-o', 
                 linewidth=2, markersize=6, label='Reject Percentage', alpha=0.8)
ax4.set_ylabel('Reject Percentage (%)', fontsize=16)
ax4.tick_params(axis='y', labelsize=16)
ax4.set_ylim(0, max(combined_reject_percentage) * 1.1 if max(combined_reject_percentage) > 0 else 100)

# Combine legends
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False, fontsize=16, ncol=2)

plt.tight_layout()
plt.savefig('/home/juchanlee/MagicDec/figure/confidence/histogram_reject_combined.png', dpi=300, bbox_inches='tight')
print("Saved: histogram_reject_combined.png")

plt.show()

# Print some statistics
print("\n=== Statistics ===")
print(f"Total draft tokens: {np.sum(draft_values)}")
print(f"Total reject tokens: {np.sum(reject_values)}")
print(f"Overall reject rate: {np.sum(reject_values) / np.sum(draft_values) * 100:.2f}%")
print(f"\nMax reject percentage (raw bins): {np.max(reject_percentage):.2f}% at bin center {bin_centers[np.argmax(reject_percentage)]:.2f}")
print(f"Max reject percentage (combined bins): {np.max(combined_reject_percentage):.2f}% at bin center {combined_bin_centers[np.argmax(combined_reject_percentage)]:.2f}")
