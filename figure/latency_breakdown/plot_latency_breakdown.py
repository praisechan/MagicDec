import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file - skip first 2 header rows, read as simple CSV
df = pd.read_csv('simulation_latency_breakdown_qwen14b_16K_new.CSV', skiprows=2, header=None)

# Extract data
batch_sizes = ['b=8', 'b=32', 'b=128']
cases = ['PIM+SD', 'PIM+SD+LB', 'PIM+SD+DV']
stages = ['Draft', 'Early-verify', 'Final-verify']
compute_types = ['GPU', 'Flash']

# Organize data into a structured format
# CSV structure: col0=batch, col1=case, col2=Draft GPU, col3=Draft Flash, 
#                col4=Early GPU, col5=Early Flash, col6=Final GPU, col7=Final Flash
data = {}

for idx, row in df.iterrows():
    # Skip empty rows
    if pd.isna(row[1]):
        continue
    
    # Get batch (if present in current row, otherwise use last batch)
    if pd.notna(row[0]):
        current_batch = row[0]
        if current_batch not in data:
            data[current_batch] = {}
    
    case = row[1]
    if current_batch in batch_sizes and case in cases:
        data[current_batch][case] = {
            'Draft': {'GPU': float(row[2]), 'Flash': float(row[3])},
            'Early-verify': {'GPU': float(row[4]), 'Flash': float(row[5])},
            'Final-verify': {'GPU': float(row[6]), 'Flash': float(row[7])}
        }

# Define colors for each stage (same color for GPU and Flash within a stage)
stage_colors = {
    'Draft': '#F4B942',           # Yellow/Gold
    'Early-verify': '#5B9BD5',    # Blue
    'Final-verify': '#70AD47'     # Green
}

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Calculate bar positions
n_batches = len(batch_sizes)
n_cases = len(cases)
bar_width = 0.6  # Width of each bar
group_gap = 2.0  # Gap between batch size groups
bar_gap = 1.0    # Gap between different cases within a batch

# Generate x positions - one position per case
case_positions = {}  # Store center positions for each case
for i, batch in enumerate(batch_sizes):
    base_x = i * (n_cases * bar_gap + group_gap)
    case_positions[batch] = []
    for j, case in enumerate(cases):
        case_center = base_x + j * bar_gap
        case_positions[batch].append(case_center)

# Plot stacked bars - combine GPU and Flash in single bar
bar_objects = {'GPU': {}, 'Flash': {}}  # Store bar objects for legend

for i, batch in enumerate(batch_sizes):
    for j, case in enumerate(cases):
        x = case_positions[batch][j]
        
        # Calculate total across all stages and compute types for this case
        total = sum(data[batch][case][stage][compute_type] 
                   for stage in stages for compute_type in compute_types)
        
        # Stack bars: for each stage, stack GPU then Flash
        bottom = 0
        for stage in stages:
            for compute_type in compute_types:
                value = data[batch][case][stage][compute_type]
                height = (value / total) * 100
                color = stage_colors[stage]
                
                # Use hatching pattern for Flash, solid for GPU
                hatch = '///' if compute_type == 'Flash' else None
                
                bar = ax.bar(x, height, bar_width, bottom=bottom, 
                           color=color, edgecolor='black', linewidth=0.8,
                           hatch=hatch)
                
                # Store bar objects for legend (only once per stage/compute_type combination)
                if batch == batch_sizes[0] and case == cases[0]:
                    if stage not in bar_objects[compute_type]:
                        bar_objects[compute_type][stage] = bar
                
                bottom += height

# Set y-axis
ax.set_ylabel('Percentage(%)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.set_yticks(range(0, 101, 10))

# Primary x-axis (case names)
case_labels = []
case_ticks = []
for batch in batch_sizes:
    case_ticks.extend(case_positions[batch])
    case_labels.extend(cases)

ax.set_xticks(case_ticks)
ax.set_xticklabels(case_labels, fontsize=11)
ax.tick_params(axis='x', which='major', length=6)

# Secondary x-axis (batch sizes)
sec = ax.secondary_xaxis(location=0)
batch_ticks = []
for batch in batch_sizes:
    batch_center = np.mean(case_positions[batch])
    batch_ticks.append(batch_center)

sec.set_xticks(batch_ticks, labels=[f'\n\n{batch}' for batch in batch_sizes])
sec.tick_params('x', length=0)
for label in sec.get_xticklabels():
    label.set_fontsize(12)
    label.set_fontweight('bold')

# Add vertical lines between batch groups - MANUAL CONTROL
sec2 = ax.secondary_xaxis(location=0)

# Manual divider line positions
# Calculate positions between batch groups automatically, but you can override these values
line_positions = []
for i in range(len(batch_sizes) + 1):
    if i == 0:
        # Left edge of first batch
        line_positions.append(case_positions[batch_sizes[0]][0] - bar_gap / 2)
    elif i == len(batch_sizes):
        # Right edge of last batch
        line_positions.append(case_positions[batch_sizes[-1]][-1] + bar_gap / 2)
    else:
        # Position between batches
        prev_batch_last = case_positions[batch_sizes[i-1]][-1]
        next_batch_first = case_positions[batch_sizes[i]][0]
        line_positions.append((prev_batch_last + next_batch_first) / 2)

# CUSTOMIZE DIVIDER LINES HERE:
# Uncomment and modify these lines to manually set divider positions
# line_positions = [-1.0, 4.5, 9.5, 14.0]  # Example: custom positions

sec2.set_xticks(line_positions, labels=[])
sec2.tick_params('x', length=60, width=1.5)

# Set x-axis limits
ax.set_xlim(line_positions[0] - 0.3, line_positions[-1] + 0.3)

# Create legend
from matplotlib.patches import Patch
legend_elements = []

# Add stage legends with GPU pattern
for stage in stages:
    legend_elements.append(Patch(facecolor=stage_colors[stage], edgecolor='black', 
                                label=f'{stage} GPU'))

# Add stage legends with Flash pattern
for stage in stages:
    legend_elements.append(Patch(facecolor=stage_colors[stage], edgecolor='black', 
                                hatch='///', label=f'{stage} Flash'))

ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), 
         ncol=3, fontsize=10)

# Add grid
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('latency_breakdown_qwen14b_16K.png', dpi=300, bbox_inches='tight')
plt.savefig('latency_breakdown_qwen14b_16K.pdf', bbox_inches='tight')

print("Figure saved as latency_breakdown_qwen14b_16K.png and .pdf")
plt.show()
