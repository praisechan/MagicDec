import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('test.csv')

# Clean up the data - forward fill case and method names
df['case'] = df.iloc[:, 0].ffill()
df['method'] = df.iloc[:, 1].ffill()
df['batch'] = df.iloc[:, 2]
df['throughput'] = df.iloc[:, 3]

# Remove rows with missing throughput data
df = df.dropna(subset=['throughput'])

# Apply abbreviations
def apply_abbreviations(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = text.replace('load balance', 'LB')
    text = text.replace('flexgen', 'flex')
    text = text.replace('2stage SD', 'SD_2')
    return text

df['method'] = df['method'].apply(apply_abbreviations)

# Create a comprehensive method label
def create_method_label(row):
    case = row['case']
    method = row['method']
    
    if pd.isna(method) or method == '':
        return case
    else:
        return method

df['method_label'] = df.apply(create_method_label, axis=1)

print("Data overview:")
print(df[['case', 'method_label', 'batch', 'throughput']])

# Create the plot
plt.figure(figsize=(16, 10))

# Group data by method
grouped = df.groupby('method_label')

# Define colors and markers
colors = plt.cm.Set1(np.linspace(0, 1, len(grouped)))
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

for i, (method, group) in enumerate(grouped):
    # Handle cases with batch sizes
    if not group['batch'].isna().all():
        # Sort by batch size for proper line connection
        group_sorted = group.sort_values('batch')
        plt.plot(group_sorted['batch'], group_sorted['throughput'], 
                marker=markers[i % len(markers)], linewidth=2, markersize=8, 
                label=method, color=colors[i], linestyle='-')
    else:
        # For cases without batch sizes, plot as scatter points
        y_values = group['throughput'].values
        x_values = np.arange(len(y_values)) + 1  # Just use sequential numbers
        plt.scatter(x_values, y_values, 
                   marker=markers[i % len(markers)], s=100, 
                   label=method, color=colors[i], alpha=0.7)

plt.xlabel('Batch Size', fontsize=14, fontweight='bold')
plt.ylabel('Throughput', fontsize=14, fontweight='bold')
plt.title('Seq.len 32K', fontsize=18, fontweight='bold', pad=20)

# Customize legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# Customize grid
plt.grid(True, alpha=0.3, linestyle='--')

# Set logarithmic scale for y-axis to better show the large range of values
plt.yscale('log')
plt.ylabel('Throughput (log scale)', fontsize=14, fontweight='bold')

# Improve tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('throughput_graph_improved.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nImproved graph saved as 'throughput_graph_improved.png'")

# Also create a bar chart version for better comparison
plt.figure(figsize=(16, 10))

# Calculate average throughput for each method
avg_throughput = df.groupby('method_label')['throughput'].mean().sort_values(ascending=True)

# Create horizontal bar chart
bars = plt.barh(range(len(avg_throughput)), avg_throughput.values)

# Color bars
colors = plt.cm.viridis(np.linspace(0, 1, len(avg_throughput)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.yticks(range(len(avg_throughput)), avg_throughput.index, fontsize=11)
plt.xlabel('Average Throughput', fontsize=14, fontweight='bold')
plt.title('Seq.len 8K - Average Throughput Comparison', fontsize=18, fontweight='bold', pad=20)

# Add value labels on bars
for i, v in enumerate(avg_throughput.values):
    plt.text(v + max(avg_throughput.values) * 0.01, i, f'{v:.1f}', 
             va='center', fontsize=10)

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()

# Save the bar chart
plt.savefig('throughput_comparison_bar.png', dpi=300, bbox_inches='tight')
plt.show()

print("Bar chart saved as 'throughput_comparison_bar.png'")
