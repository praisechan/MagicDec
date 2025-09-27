#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Read the CSV data
prefix_len = 32800
csv_path = f"/home/juchanlee/MagicDec/figure/confidence/data/wrong_data/run2step_Meta-Llama-3.1-8B_pg19_histogram_data_{prefix_len}.csv"
df = pd.read_csv(csv_path)

print("All experiment names in the CSV:")
for i, exp in enumerate(df['experiment']):
    print(f"{i}: {exp}")

print("\n" + "="*80)

# Find baseline reject row (ends with 'report_reject' but not 'budget_X.XX_reject')
baseline_rows = df[df['experiment'].str.contains('report_reject$') & ~df['experiment'].str.contains('budget_[0-9]')]
print(f"Baseline rows found: {len(baseline_rows)}")
if not baseline_rows.empty:
    baseline_row = baseline_rows.iloc[0]
    print(f"Baseline experiment name: {baseline_row['experiment']}")
    
    # Get the bin columns (exclude 'experiment' column)
    bin_columns = [col for col in df.columns if col != 'experiment']
    print(f"Total bins: {len(bin_columns)}")
    
    # Show first 10 bins of baseline data
    print("\nFirst 10 bins of baseline data:")
    for i in range(10):
        print(f"{bin_columns[i]}: {baseline_row[bin_columns[i]]}")
    
    # Calculate 0.0-0.1 bin (first 5 bins: 0.00-0.02, 0.02-0.04, 0.04-0.06, 0.06-0.08, 0.08-0.10)
    first_5_bins = baseline_row[bin_columns[0:5]].values
    print(f"\nFirst 5 bins values: {first_5_bins}")
    print(f"Sum of first 5 bins (0.0-0.1): {np.sum(first_5_bins)}")
    
    # Show total for baseline
    print(f"Total baseline reject tokens: {np.sum(baseline_row[bin_columns])}")

print("\n" + "="*80)

# Check all reject rows
all_reject_rows = df[df['experiment'].str.contains('reject')]
print("All reject experiment names:")
for i, exp in enumerate(all_reject_rows['experiment']):
    print(f"{i}: {exp}")