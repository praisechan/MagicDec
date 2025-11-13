#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Read the CSV data
prefix_len = 32800
csv_path = f"/home/juchanlee/MagicDec/figure/confidence/data/wrong_data/run2step_Meta-Llama-3.1-8B_pg19_histogram_data_{prefix_len}.csv"
df = pd.read_csv(csv_path)

print("DataFrame shape:", df.shape)
print("DataFrame contents:")
print(df)

print("\n" + "="*80)

# Extract reject data for different budgets
reject_rows = {}

# Find baseline reject row (ends with 'report_reject' but not 'budget_X.XX_reject')
baseline_rows = df[df['experiment'].str.contains('report_reject$') & ~df['experiment'].str.contains('budget_[0-9]')]
print(f"Baseline rows found: {len(baseline_rows)}")
if not baseline_rows.empty:
    reject_rows['baseline'] = baseline_rows.iloc[0]
    print(f"Baseline experiment: {reject_rows['baseline']['experiment']}")

# Extract budget-specific reject data
budget_values = ['0.10', '0.25', '0.40']
for budget in budget_values:
    budget_rows = df[df['experiment'].str.contains(f'budget_{budget}_reject')]
    if not budget_rows.empty:
        reject_rows[f'budget_{budget}'] = budget_rows.iloc[0]
        print(f"Budget {budget} experiment: {reject_rows[f'budget_{budget}']['experiment']}")

print("\n" + "="*50 + " DETAILED ANALYSIS " + "="*50)

# Get the bin columns (exclude 'experiment' column)
bin_columns = [col for col in df.columns if col != 'experiment']
print(f"Total bin columns: {len(bin_columns)}")
print(f"First 10 bin columns: {bin_columns[:10]}")

# Extract counts for all reject types
reject_counts = {}
for key, row in reject_rows.items():
    reject_counts[key] = row[bin_columns].values
    print(f"\n{key.upper()} REJECT DATA:")
    print(f"  Total tokens: {np.sum(reject_counts[key])}")
    print(f"  First 10 bin values: {reject_counts[key][:10]}")
    print(f"  First 5 bin values (0.0-0.1): {reject_counts[key][:5]}")
    print(f"  Sum of first 5 bins: {np.sum(reject_counts[key][:5])}")

print("\n" + "="*80)
print("Now let's see what happens when we group bins into 0.1 width intervals...")

# Group bins into 0.1 width intervals (combine 5 bins of 0.02 width each)
bins_per_group = 5

for key in reject_counts.keys():
    print(f"\n{key.upper()} - 0.1 WIDTH GROUPING:")
    combined_counts = []
    
    for i in range(0, len(bin_columns), bins_per_group):
        # Determine the range for this group
        end_idx = min(i + bins_per_group, len(bin_columns))
        
        # Sum counts from all bins in this group
        combined_count = np.sum(reject_counts[key][i:end_idx])
        combined_counts.append(combined_count)
        
        # Create label for this group
        start_bin = bin_columns[i]
        end_bin = bin_columns[end_idx - 1]
        start_val = float(start_bin.split('-')[0])
        end_val = float(end_bin.split('-')[1])
        label = f"{start_val:.1f}-{end_val:.1f}"
        
        print(f"  {label}: {combined_count} (from bins {i} to {end_idx-1}: {reject_counts[key][i:end_idx]})")
        
        if i >= 15:  # Only show first few groups
            break
    
    print(f"  Total after grouping: {np.sum(combined_counts)}")
    print(f"  Original total: {np.sum(reject_counts[key])}")