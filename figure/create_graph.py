import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('test.csv')

# Clean up the data
# Forward fill the case names and method names
df['case'] = df.iloc[:, 0].fillna(method='ffill')
df['method'] = df.iloc[:, 1].fillna(method='ffill')
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

# Create a more readable method name by combining case and method info
def create_method_label(row):
    case = row['case']
    method = row['method']
    
    if pd.isna(method) or method == '':
        return case
    else:
        return f"{case}: {method}"

df['method_label'] = df.apply(create_method_label, axis=1)

# Filter data into two groups: cases with batch sizes and cases without
df_with_batch = df[df['batch'].notna()].copy()
df_without_batch = df[df['batch'].isna()].copy()

# Set up the plot style
plt.figure(figsize=(18, 10))
sns.set_style("whitegrid")

# Get unique cases with batch sizes and batch sizes
unique_cases_with_batch = df_with_batch['case'].unique()
batch_sizes = sorted(df_with_batch['batch'].unique())

# Get unique cases without batch sizes (CPU cases)
unique_cases_without_batch = df_without_batch['case'].unique()

# Calculate total number of groups for x-axis positioning
total_groups = len(unique_cases_with_batch) + len(unique_cases_without_batch)

# Set up bar positions
x_with_batch = np.arange(len(unique_cases_with_batch))
x_without_batch = np.arange(len(unique_cases_with_batch), total_groups)
width = 0.25  # Width of each bar
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colors for batch sizes 4, 16, 64

# Create grouped bars for cases with batch sizes
for i, batch_size in enumerate(batch_sizes):
    batch_data = df_with_batch[df_with_batch['batch'] == batch_size]
    throughputs = []
    
    for case in unique_cases_with_batch:
        case_data = batch_data[batch_data['case'] == case]
        if not case_data.empty:
            throughputs.append(case_data['throughput'].iloc[0])
        else:
            throughputs.append(0)  # If no data for this case/batch combination
    
    plt.bar(x_with_batch + i * width, throughputs, width, 
           label=f'Batch {int(batch_size)}', color=colors[i], alpha=0.8)

# Create single bars for CPU cases (cases without batch sizes)
cpu_color = '#d62728'  # Red color for CPU cases
for j, case in enumerate(unique_cases_without_batch):
    case_data = df_without_batch[df_without_batch['case'] == case]
    avg_throughput = case_data['throughput'].mean()  # Average the multiple readings
    
    plt.bar(x_without_batch[j], avg_throughput, width * 3, 
           label=f'CPU (avg)' if j == 0 else "", color=cpu_color, alpha=0.8)

# Customize the plot
plt.xlabel('Cases', fontsize=18, fontweight='bold')
plt.ylabel('Throughput', fontsize=18, fontweight='bold')
plt.title('Seq.len 32K', fontsize=24, fontweight='bold')

# Set x-axis labels
case_labels = []

# Add labels for cases with batch sizes
for case in unique_cases_with_batch:
    case_method = df_with_batch[df_with_batch['case'] == case]['method'].iloc[0]
    case_labels.append(case_method)

# Add labels for CPU cases
for case in unique_cases_without_batch:
    case_method = df_without_batch[df_without_batch['case'] == case]['method'].iloc[0]
    case_labels.append(case_method)

# Set x-tick positions (middle of grouped bars for batch cases, center for CPU cases)
x_tick_positions = list(x_with_batch + width) + list(x_without_batch)
plt.xticks(x_tick_positions, case_labels, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.grid(True, alpha=0.3, axis='y')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('throughput_bar_graph.png', dpi=300, bbox_inches='tight')
plt.show()

print("Grouped bar graph saved as 'throughput_bar_graph.png'")
print("\nData summary:")
print(df.groupby('method_label')[['batch', 'throughput']].describe())
