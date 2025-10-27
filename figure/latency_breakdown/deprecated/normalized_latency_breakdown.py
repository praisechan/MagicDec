import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set font to DejaVu Sans (clean sans-serif font similar to Arial)
plt.rcParams['font.family'] = 'DejaVu Sans'

def load_data_from_csv(csv_file='simulation_latency_breakdown_qwen14b_16K.CSV'):
    """Load latency data from CSV file and convert to the expected format"""
    
    # Read the CSV file
    df = pd.read_csv(csv_file, header=[0, 1])
    
    # Clean up the dataframe - remove empty rows and fix column names
    df = df.dropna(how='all')
    
    # The structure should be: [draft_gpu, draft_flash, verify_gpu, verify_flash, settle_gpu, settle_flash]
    data = {}
    
    current_config = None
    for idx, row in df.iterrows():
        # Check if this row contains a configuration name
        first_col = str(row.iloc[0]).strip()
        if first_col in ['PIM+SD', 'PIM+SD+LB', 'PIM+SD+LB+DV']:
            current_config = first_col
            data[current_config] = {}
        
        # Check if this row contains batch data
        if current_config and 'batch' in str(row.iloc[1]).lower():
            batch_str = str(row.iloc[1]).strip()
            if 'batch32' in batch_str:
                batch_name = 'Batch 32'
            elif 'batch 128' in batch_str:
                batch_name = 'Batch 128'
            elif 'batch 8' in batch_str:
                batch_name = 'Batch 8'
            else:
                continue
            
            # Extract the latency values: [draft_gpu, draft_flash, verify_gpu, verify_flash, settle_gpu, settle_flash]
            try:
                draft_gpu = float(row.iloc[2])
                draft_flash = float(row.iloc[3])
                verify_gpu = float(row.iloc[4])
                verify_flash = float(row.iloc[5])
                settle_gpu = float(row.iloc[6])
                settle_flash = float(row.iloc[7])
                
                data[current_config][batch_name] = [
                    draft_gpu, draft_flash, verify_gpu, verify_flash, settle_gpu, settle_flash
                ]
            except (ValueError, IndexError):
                continue
    
    return data

def create_normalized_breakdown():
    """Create a normalized latency breakdown figure where each batch's PIM+SD is set to 1.0"""
    
    # Load data from CSV file
    original_data = load_data_from_csv()
    
    # Calculate normalization factors (total latency of PIM+SD for each batch)
    configs = list(original_data.keys())
    batch_sizes = list(original_data[configs[0]].keys())
    
    normalization_factors = {}
    for batch in batch_sizes:
        total_latency = sum(original_data['PIM+SD'][batch])
        normalization_factors[batch] = total_latency
    
    # Normalize the data
    normalized_data = {}
    for config in configs:
        normalized_data[config] = {}
        for batch in batch_sizes:
            normalized_values = []
            for component_value in original_data[config][batch]:
                normalized_value = component_value / normalization_factors[batch]
                normalized_values.append(normalized_value)
            normalized_data[config][batch] = normalized_values
    
    # Academic style figure setup
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjusted for horizontal layout
    
    # Academic color palette - professional and colorblind-friendly
    stage_colors = {
        'draft': '#E8E8E8',     # Light gray
        'verify': '#A8A8A8',    # Medium gray
        'settle': '#6B6B6B'     # Dark gray
    }
    
    # Academic patterns for GPU vs Flash
    gpu_pattern = None      # Solid fill for GPU
    flash_pattern = '...'   # Subtle dots for Flash (academic style)
    
    # Academic bar properties
    bar_height = 0.22  # Bar height for horizontal bars
    edge_linewidth = 0.8  # Thinner, more subtle edges
    edge_color = '#333333'  # Dark gray instead of pure black
    y_positions = []
    
    # Calculate positions - group by batch size, then by configuration
    for i, batch in enumerate(batch_sizes):
        batch_center = i * 1.2  # spacing between batch groups
        for j, config in enumerate(configs):
            y_pos = batch_center - bar_height + j * (bar_height + 0.05)
            y_positions.append(y_pos)
    
    # Component labels and their properties
    components = [
        ('draft_gpu', 'Draft (GPU)', stage_colors['draft'], gpu_pattern),
        ('draft_flash', 'Draft (Flash)', stage_colors['draft'], flash_pattern),
        ('verify_gpu', 'Early Verify (GPU)', stage_colors['verify'], gpu_pattern),
        ('verify_flash', 'Early Verify (Flash)', stage_colors['verify'], flash_pattern),
        ('settle_gpu', 'Final Verify (GPU)', stage_colors['settle'], gpu_pattern),
        ('settle_flash', 'Final Verify (Flash)', stage_colors['settle'], flash_pattern)
    ]
    
    # Prepare stacked data
    lefts = [0] * len(y_positions)
    
    # Track which labels we've added to legend to avoid duplicates
    legend_added = set()
    
    for component_idx, (comp_key, comp_label, color, pattern) in enumerate(components):
        values = []
        
        # Extract normalized values for this component - grouped by batch size first
        for batch in batch_sizes:
            for config in configs:
                values.append(normalized_data[config][batch][component_idx])
        
        # Create the horizontal bar - academic style
        bars = ax.barh(y_positions, values, bar_height, left=lefts,
                      color=color, hatch=pattern, alpha=0.9, 
                      edgecolor=edge_color, linewidth=edge_linewidth)
        
        # Add to legend only if not already added
        if comp_label not in legend_added:
            bars[0].set_label(comp_label)
            legend_added.add(comp_label)
        
        # Update lefts for next layer
        lefts = [l + v for l, v in zip(lefts, values)]
    
    # Customize plot - academic style
    ax.set_xlabel('Normalized Latency', fontsize=12, fontweight='normal')
    ax.set_xlim(0, max([max(lefts)] + [0]) * 1.05)
    
    # Set up hierarchical y-axis labels - academic style
    ax.set_yticks(y_positions)
    # Configuration names as y-tick labels
    config_labels = ['PIM+SD', 'PIM+SD+LB', 'PIM+SD+LB+DV']
    ax.set_yticklabels(config_labels * len(batch_sizes), fontsize=10, rotation=0)
    
    # Academic-style tick parameters
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=10, length=3)
    
    # Create second y-axis for batch size labels
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    
    # Calculate group centers for batch labels
    group_centers = [i * 1.2 for i in range(len(batch_sizes))]
    batch_labels = ['8', '32', '128']
    
    # Set batch size labels on the second y-axis (left)
    ax2.set_yticks(group_centers)
    ax2.set_yticklabels([f'Batch {label}' for label in batch_labels], fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=11, length=0, pad=20)
    ax2.yaxis.set_ticks_position('left')
    ax2.yaxis.set_label_position('left')
    
    # Move the second y-axis labels to the left of the main axis
    ax2.spines['left'].set_position(('outward', 30))
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    
    # Add horizontal divider lines between batch groups
    x_max = max([max(lefts)] + [0]) * 1.05
    for i in range(1, len(batch_sizes)):
        sep_pos = (group_centers[i-1] + group_centers[i]) / 2
        ax.axhline(y=sep_pos, color='#666666', linestyle='-', alpha=0.6, linewidth=1.0, 
                  xmin=0, xmax=1, clip_on=False)
    
    # Create custom legend - academic style
    legend_elements = []
    
    # Add stage colors with custom labels
    stage_labels = {
        'draft': 'Draft Stage',
        'verify': 'Early Verify Stage', 
        'settle': 'Final Verify Stage'
    }
    for stage, color in stage_colors.items():
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.9, 
                                           edgecolor=edge_color, linewidth=edge_linewidth,
                                           label=stage_labels[stage]))
    
    # Add memory type patterns
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='#A8A8A8', alpha=0.9, 
                                       hatch=gpu_pattern, edgecolor=edge_color, 
                                       linewidth=edge_linewidth, label='GPU'))
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='#A8A8A8', alpha=0.9, 
                                       hatch=flash_pattern, edgecolor=edge_color,
                                       linewidth=edge_linewidth, label='Flash'))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=10, 
              frameon=False, columnspacing=1.0, handletextpad=0.5, ncol=3)
    
    # Academic-style grid - more subtle
    ax.grid(True, alpha=0.2, axis='x', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Adjust layout with extra left margin for hierarchical labels - academic style
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.25, right=0.85, top=0.95)
    
    # Save the figure - academic quality
    plt.savefig('normalized_latency_breakdown.png', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='png')
    plt.savefig('normalized_latency_breakdown.pdf', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='pdf')
    
    print("Academic-style normalized figure saved as normalized_latency_breakdown.png and .pdf")
    
    # Print normalization summary
    print("\nNormalization Summary:")
    for batch in batch_sizes:
        print(f"{batch}: {normalization_factors[batch]:.1f}ms → 1.0")
        for config in configs:
            total = sum(normalized_data[config][batch])
            if config == 'PIM+SD':
                print(f"  {config}: 1.000 (baseline)")
            else:
                improvement = (1.0 - total) * 100
                print(f"  {config}: {total:.3f} ({improvement:+.1f}%)")
        print()
    
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    print("\nCreating normalized latency breakdown figure...")
    create_normalized_breakdown()
