import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('dynamic_verify.csv')

# Clean the data - fill forward the prefix_len values
df['prefix_len'] = df['prefix_len'].fillna(method='ffill')

# Remove rows with missing data
df = df.dropna()

# Separate data for 8K and 16K
data_8k = df[df['prefix_len'] == '8K']
data_16k = df[df['prefix_len'] == '16K']

# Set global font sizes
plt.rcParams.update({'font.size': 14})

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Set up the bar positions
gamma_values = [5, 4, 3]
x_pos = np.arange(len(gamma_values))
width = 0.35

# Plot 8K data
ax1.bar(x_pos - width/2, data_8k['dynamic'].values, width, label='Dynamic', color='skyblue', alpha=0.8)
ax1.bar(x_pos + width/2, data_8k['baseline(no dynamic)'].values, width, label='Baseline (no dynamic)', color='lightcoral', alpha=0.8)

ax1.set_xlabel('Gamma1 Values', fontsize=16)
ax1.set_ylabel('Latency', fontsize=16)
ax1.set_title('8K Prefix Length', fontsize=18)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(gamma_values, fontsize=14)
ax1.legend(fontsize=14)
ax1.grid(True, alpha=0.3)

# Add value labels on bars for 8K
for i, (dynamic, baseline) in enumerate(zip(data_8k['dynamic'].values, data_8k['baseline(no dynamic)'].values)):
    ax1.text(i - width/2, dynamic + 1, f'{dynamic}', ha='center', va='bottom', fontsize=12)
    ax1.text(i + width/2, baseline + 1, f'{baseline}', ha='center', va='bottom', fontsize=12)

# Plot 16K data
ax2.bar(x_pos - width/2, data_16k['dynamic'].values, width, label='Dynamic', color='skyblue', alpha=0.8)
ax2.bar(x_pos + width/2, data_16k['baseline(no dynamic)'].values, width, label='Baseline (no dynamic)', color='lightcoral', alpha=0.8)

ax2.set_xlabel('Gamma1 Values', fontsize=16)
ax2.set_ylabel('Latency', fontsize=16)
ax2.set_title('16K Prefix Length', fontsize=18)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(gamma_values, fontsize=14)
ax2.legend(fontsize=14)
ax2.grid(True, alpha=0.3)

# Add value labels on bars for 16K
for i, (dynamic, baseline) in enumerate(zip(data_16k['dynamic'].values, data_16k['baseline(no dynamic)'].values)):
    ax2.text(i - width/2, dynamic + 1, f'{dynamic}', ha='center', va='bottom', fontsize=12)
    ax2.text(i + width/2, baseline + 1, f'{baseline}', ha='center', va='bottom', fontsize=12)

# Adjust layout and save
plt.tight_layout()
plt.suptitle('Dynamic vs Baseline Performance Comparison', y=1.02, fontsize=20)
plt.savefig('dynamic_verify_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph saved as 'dynamic_verify_comparison.png'")
