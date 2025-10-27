import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Read the CSV file with multi-level header
df = pd.read_csv('simulation_latency_breakdown_qwen14b_16K.CSV', header=[0, 1])

# Clean up the dataframe
df.columns = df.columns.map(lambda x: (x[0].strip() if x[0] else '', x[1].strip() if x[1] else ''))

# Remove empty rows and reset columns
# First two columns are case name and batch size
df_clean = df.dropna(subset=[(df.columns[1])])

# Rename columns for easier access
columns_renamed = ['Case', 'Batch', 'draft_GPU', 'draft_Flash', 'verify1_GPU', 
                   'verify1_Flash', 'settle_GPU', 'settle_Flash', 'Total']
df_clean.columns = columns_renamed

# Fill case names forward
df_clean.loc[:, 'Case'] = df_clean['Case'].ffill()

# Remove rows with empty batch
df_clean = df_clean[df_clean['Batch'].notna()]

# Convert batch names to consistent format
df_clean['Batch'] = df_clean['Batch'].str.strip()
df_clean['Batch'] = df_clean['Batch'].replace({'batch32': 'batch 32'})

# Convert numeric columns to float
numeric_cols = ['draft_GPU', 'draft_Flash', 'verify1_GPU', 'verify1_Flash', 
                'settle_GPU', 'settle_Flash', 'Total']
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

print("Data loaded:")
print(df_clean)

# Define the order of batches and cases
batch_order = ['batch 8', 'batch 32', 'batch 128']
cases = df_clean['Case'].unique()

# Define colors for each stage (base colors)
# Draft: blue shades, Verify1: orange shades, Settle: green shades
colors = {
    'draft_GPU': '#1f77b4',      # solid blue
    'draft_Flash': '#1f77b4',    # blue with hatch
    'verify1_GPU': '#ff7f0e',    # solid orange
    'verify1_Flash': '#ff7f0e',  # orange with hatch
    'settle_GPU': '#2ca02c',     # solid green
    'settle_Flash': '#2ca02c',   # green with hatch
}

# Define hatches (solid for GPU, hatched for Flash)
hatches = {
    'draft_GPU': '',
    'draft_Flash': '///',
    'verify1_GPU': '',
    'verify1_Flash': '///',
    'settle_GPU': '',
    'settle_Flash': '///',
}

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 6))

# Set up bar positioning
num_batches = len(batch_order)
num_cases = len(cases)
bar_width = 0.25
group_width = num_cases * bar_width
group_spacing = 0.3

# Stack order for the bars
stack_order = ['draft_GPU', 'draft_Flash', 'verify1_GPU', 'verify1_Flash', 
               'settle_GPU', 'settle_Flash']

# Plot bars
batch_positions = []
batch_labels = []

for batch_idx, batch in enumerate(batch_order):
    # Calculate the center position for this batch group
    group_center = batch_idx * (group_width + group_spacing)
    
    for case_idx, case in enumerate(cases):
        # Get data for this case and batch
        row = df_clean[(df_clean['Case'] == case) & (df_clean['Batch'] == batch)]
        
        if row.empty:
            continue
        
        # Calculate x position for this bar
        x_pos = group_center + case_idx * bar_width
        
        # Plot stacked bars
        bottom = 0
        for component in stack_order:
            value = row[component].values[0]
            
            bar = ax.bar(x_pos, value, bar_width, bottom=bottom,
                        color=colors[component], edgecolor='black', linewidth=0.5,
                        hatch=hatches[component], label=component if batch_idx == 0 and case_idx == 0 else "")
            bottom += value
    
    # Store batch group center position for labeling
    batch_center = group_center + (num_cases - 1) * bar_width / 2
    batch_positions.append(batch_center)
    batch_labels.append(batch)

# Set x-axis labels (cases) at each bar position
case_positions = []
case_tick_labels = []
for batch_idx in range(num_batches):
    group_center = batch_idx * (group_width + group_spacing)
    for case_idx, case in enumerate(cases):
        x_pos = group_center + case_idx * bar_width
        case_positions.append(x_pos)
        case_tick_labels.append(case)

ax.set_xticks(case_positions)
ax.set_xticklabels(case_tick_labels, rotation=45, ha='right')

# Add batch size labels as secondary labels
for pos, label in zip(batch_positions, batch_labels):
    ax.text(pos, -0.15, label, transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=11, fontweight='bold')

# Labels and title
ax.set_ylabel('Latency (ms)', fontsize=12)
ax.set_xlabel('Cases', fontsize=12)
ax.set_title('Latency Breakdown by Configuration and Batch Size', fontsize=14, fontweight='bold', pad=20)

# Create custom legend
legend_elements = [
    mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label='Draft GPU'),
    mpatches.Patch(facecolor='#1f77b4', edgecolor='black', hatch='///', label='Draft Flash'),
    mpatches.Patch(facecolor='#ff7f0e', edgecolor='black', label='Verify1 GPU'),
    mpatches.Patch(facecolor='#ff7f0e', edgecolor='black', hatch='///', label='Verify1 Flash'),
    mpatches.Patch(facecolor='#2ca02c', edgecolor='black', label='Settle GPU'),
    mpatches.Patch(facecolor='#2ca02c', edgecolor='black', hatch='///', label='Settle Flash'),
]

ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=10)

# Grid
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save and show
plt.savefig('latency_breakdown_stacked.png', dpi=300, bbox_inches='tight')
plt.savefig('latency_breakdown_stacked.pdf', bbox_inches='tight')
print("\nFigure saved as 'latency_breakdown_stacked.png' and 'latency_breakdown_stacked.pdf'")

plt.show()
