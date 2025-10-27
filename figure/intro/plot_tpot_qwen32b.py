import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('tpot_qwen32b.CSV', skiprows=1)
df = df.iloc[:, :4]  # Keep only the first 4 columns
df.columns = ['Batch Size', '8K', '16K', '32K']

# Set up the figure
fig, ax = plt.subplots(figsize=(8, 6))

# Parameters for the bar chart
sequence_lengths = ['8K', '16K', '32K']
batch_sizes = df['Batch Size'].values
n_sequences = len(sequence_lengths)
n_batches = len(batch_sizes)

# Width of bars and spacing
bar_width = 0.25
group_spacing = 0.1
group_width = n_batches * bar_width + group_spacing

# X positions for each group
x_positions = np.arange(n_sequences) * (group_width + 0.5)

# Colors for different batch sizes - blue with decreasing brightness
colors = ["#9dbbf8", "#1352cf", "#163880"]  # Light blue, medium blue, dark blue
hatches = ['///', '...', '---']

# Plot bars for each batch size
bars = []
for i, batch_size in enumerate(batch_sizes):
    positions = x_positions + i * bar_width
    values = df.loc[df['Batch Size'] == batch_size, sequence_lengths].values.flatten()
    bar = ax.bar(positions, values, bar_width, 
                 label=f'b={int(batch_size)}',
                 color=colors[i], 
                 edgecolor='black',
                 linewidth=1,
                 alpha=1.0,
                 zorder=3)
    bars.append(bar)

# Set y-axis to log scale
ax.set_yscale('log')

# Set y-axis range (lower limit raised to make 10^0 appear higher)
ax.set_ylim(bottom=0.2, top=400)

# Set labels and title
ax.set_ylabel('TPOT (sec/token)', fontsize=24)
ax.set_xlabel('Context Length', fontsize=24)
# ax.set_ylabel('TPOT (token/sec)', fontsize=24, fontweight='bold')
# ax.set_xlabel('Context Length', fontsize=24, fontweight='bold')

# Set x-ticks at the center of each group
ax.set_xticks(x_positions + bar_width)
ax.set_xticklabels(sequence_lengths, fontsize=24)
ax.tick_params(axis='y', labelsize=24)

# Add grid for better readability
ax.grid(axis='y', alpha=0.5, linestyle='-')
ax.set_axisbelow(True)

# Add legend above the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), 
          ncol=3, frameon=False, fontsize=21, 
          edgecolor='black', fancybox=False,
          handlelength=1, handleheight=1)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('tpot_qwen32b.png', dpi=300, bbox_inches='tight')
plt.savefig('tpot_qwen32b.pdf', bbox_inches='tight')
print("Figure saved as tpot_qwen32b.png and tpot_qwen32b.pdf")

# Show the plot
plt.show()
