import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data
df = pd.read_csv('kv_cache_size.csv')

# Extract context lengths and sizes
context_lengths = df.columns[1:].tolist()  # Skip first column (unnamed index column)
sizes_gb = df.iloc[0, 1:].values  # Get size values from first data row

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Create bar positions
positions = np.arange(len(context_lengths))
bar_width = 0.4

# Plot bars
bars = ax.bar(positions, sizes_gb, width=bar_width, color="#FFC000",  # Medium blue
              edgecolor='black', linewidth=1)

# Set up axes
ax.set_ylabel('KV Cache Size (GB)', fontsize=24)
ax.set_xlabel('Context Length', fontsize=24)
ax.set_xticks(positions)
ax.set_xticklabels(context_lengths, fontsize=24)
ax.tick_params(axis='y', labelsize=24)

# # Add value labels on top of bars
# for i, (bar, value) in enumerate(zip(bars, sizes_gb)):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height,
#             f'{int(value)} GB',
#             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Grid
ax.grid(axis='y', alpha=0.5, linestyle='-')
ax.set_axisbelow(True)

# Set y-axis to start from 0
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('kv_cache_size_figure.png', dpi=300, bbox_inches='tight')
print("KV cache size figure saved as 'kv_cache_size_figure.png'")
plt.close()

print("\nData Summary:")
print("=" * 50)
print("\nKV Cache Size (GB) by Context Length:")
for context, size in zip(context_lengths, sizes_gb):
    print(f"  {context}: {int(size)} GB")
