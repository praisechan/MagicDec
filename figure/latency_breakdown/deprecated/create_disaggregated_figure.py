import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def create_disaggregated_latency_figure():
    """Create a latency breakdown figure with separate colors for each stage and patterns for GPU/Flash"""
    
    # Load data from CSV file
    data = load_data_from_csv()
    
    fig, ax = plt.subplots(figsize=(12, 12))  # Reduced width for compact layout
    
    # Colors for different stages - more distinct colors
    stage_colors = {
        'draft': '#FF6B6B',     # Red/coral
        'verify': '#4ECDC4',    # Teal/cyan  
        'settle': '#FFD93D'     # Yellow/gold - distinct from verify
    }
    
    # Patterns for GPU vs Flash
    gpu_pattern = None      # Solid fill for GPU
    flash_pattern = '///'   # Diagonal lines for Flash
    
    configs = list(data.keys())
    batch_sizes = list(data[configs[0]].keys())
    
    bar_width = 0.3  # Increased bar width
    x_positions = []
    
    # Calculate positions - 3 bars per group (one for each batch size) with minimal spacing
    for i, config in enumerate(configs):
        group_center = i * 1.2  # Much smaller spacing between groups
        for j, batch in enumerate(batch_sizes):
            x_pos = group_center - bar_width + j * (bar_width + 0.05)  # Minimal gap between bars
            x_positions.append(x_pos)
    
    # Component labels and their properties
    components = [
        ('draft_gpu', 'Draft (GPU)', stage_colors['draft'], gpu_pattern),
        ('draft_flash', 'Draft (Flash)', stage_colors['draft'], flash_pattern),
        ('verify_gpu', 'Verify (GPU)', stage_colors['verify'], gpu_pattern),
        ('verify_flash', 'Verify (Flash)', stage_colors['verify'], flash_pattern),
        ('settle_gpu', 'Settle (GPU)', stage_colors['settle'], gpu_pattern),
        ('settle_flash', 'Settle (Flash)', stage_colors['settle'], flash_pattern)
    ]
    
    # Prepare stacked data
    bottoms = [0] * len(x_positions)
    
    # Track which labels we've added to legend to avoid duplicates
    legend_added = set()
    
    for component_idx, (comp_key, comp_label, color, pattern) in enumerate(components):
        values = []
        
        # Extract values for this component across all configurations and batch sizes
        for config in configs:
            for batch in batch_sizes:
                values.append(data[config][batch][component_idx])
        
        # Create the bar
        bars = ax.bar(x_positions, values, bar_width, bottom=bottoms,
                     color=color, hatch=pattern, alpha=0.8, 
                     edgecolor='white', linewidth=0.5)
        
        # Add to legend only if not already added
        if comp_label not in legend_added:
            bars[0].set_label(comp_label)
            legend_added.add(comp_label)
        
        # Update bottoms for next layer
        bottoms = [b + v for b, v in zip(bottoms, values)]
    
    # Customize plot
    ax.set_ylabel('Latency (ms)', fontsize=16, fontweight='bold')
    ax.set_title('Latency Breakdown (Seq.len 16K)', fontsize=18, fontweight='bold', pad=20)
    # Use linear scale instead of log scale
    ax.set_ylim(0, max([max(bottoms)] + [0]) * 1.1)
    
    # Set x-axis with two-level indexing
    ax.set_xticks(x_positions)
    # Remove "batch" and use only numbers
    batch_labels = ['8', '32', '128']
    ax.set_xticklabels(batch_labels * len(configs), fontsize=12)
    
    # Add configuration labels below x-axis (non-overlapping)
    group_centers = [i * 1.2 for i in range(len(configs))]  # Match the minimal spacing
    for i, (config, pos) in enumerate(zip(configs, group_centers)):
        # Position configuration labels lower to avoid overlap with batch numbers
        ax.text(pos, -max([max(bottoms)] + [0]) * 0.15, config, ha='center', va='top', 
                fontsize=14, fontweight='bold')
    
    # Add lines to separate configuration groups
    for i in range(1, len(configs)):
        sep_pos = (group_centers[i-1] + group_centers[i]) / 2
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)
    
    # Create custom legend
    legend_elements = []
    
    # Add stage colors with custom labels
    stage_labels = {
        'draft': 'Draft Stage',
        'verify': 'Early Verify Stage', 
        'settle': 'Final Verify Stage'
    }
    for stage, color in stage_colors.items():
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, 
                                           label=stage_labels[stage]))
    
    # Add separator
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='none', edgecolor='none', label=''))
    
    # Add memory type patterns
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.8, 
                                       hatch=gpu_pattern, label='GPU'))
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.8, 
                                       hatch=flash_pattern, label='Flash'))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, 
              frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout with extra bottom margin for two-level labels
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    # Save the figure
    plt.savefig('disaggregated_latency_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig('disaggregated_latency_breakdown.pdf', dpi=300, bbox_inches='tight')
    
    print("Disaggregated figure saved as disaggregated_latency_breakdown.png and .pdf")
    plt.show()
    
    return fig, ax

def create_side_by_side_breakdown():
    """Alternative version with side-by-side bars for GPU and Flash within each stage"""
    
    # Load data from CSV file
    data = load_data_from_csv()
    
    fig, ax = plt.subplots(figsize=(14, 12))  # Reduced width for compact layout
    
    # Colors for different stages - same as stacked version
    stage_colors = {
        'draft': '#FF6B6B',     # Red/coral
        'verify': '#4ECDC4',    # Teal/cyan  
        'settle': '#FFD93D'     # Yellow/gold - distinct from verify
    }
    
    configs = list(data.keys())
    batch_sizes = list(data[configs[0]].keys())
    
    bar_width = 0.15  # Wider bars
    spacing = 0.01    # Minimal spacing between GPU and Flash bars
    
    x_positions = []
    labels = []
    
    # Calculate positions with minimal spacing
    for i, config in enumerate(configs):
        group_center = i * 2.0  # Much smaller spacing between groups
        for j, batch in enumerate(batch_sizes):
            # Position for this batch within the group
            batch_center = group_center - 0.3 + j * 0.3  # Closer batch spacing
            x_positions.append(batch_center)
            labels.append(batch)
    
    # For each position, we'll have 6 bars (3 stages × 2 memory types)
    all_bars = []
    component_names = ['Draft\n(GPU)', 'Draft\n(Flash)', 'Verify\n(GPU)', 
                      'Verify\n(Flash)', 'Settle\n(GPU)', 'Settle\n(Flash)']
    
    # Stage labels for legend
    stage_labels = {
        'draft': 'Draft',
        'verify': 'Early Verify', 
        'settle': 'Final Verify'
    }
    
    for comp_idx in range(6):
        stage = ['draft', 'draft', 'verify', 'verify', 'settle', 'settle'][comp_idx]
        memory_type = ['gpu', 'flash'][comp_idx % 2]
        
        values = []
        bar_positions = []
        
        for pos_idx, config in enumerate(configs):
            for batch in batch_sizes:
                values.append(data[config][batch][comp_idx])
                # Calculate exact position for this bar
                base_pos = x_positions[pos_idx * len(batch_sizes) + batch_sizes.index(batch)]
                bar_pos = base_pos - 3*bar_width - 2*spacing + comp_idx * (bar_width + spacing/3)
                bar_positions.append(bar_pos)
        
        # Create bars
        if memory_type == 'gpu':
            bars = ax.bar(bar_positions, values, bar_width, 
                         color=stage_colors[stage], alpha=0.9,
                         label=f'{stage_labels[stage]} (GPU)' if comp_idx < 2 or comp_idx == 2 or comp_idx == 4 else "")
        else:
            bars = ax.bar(bar_positions, values, bar_width, 
                         color=stage_colors[stage], alpha=0.6, hatch='///',
                         label=f'{stage_labels[stage]} (Flash)' if comp_idx < 2 or comp_idx == 3 or comp_idx == 5 else "")
        
        all_bars.append(bars)
    
    # Customize plot
    ax.set_ylabel('Latency (ms)', fontsize=16, fontweight='bold')
    ax.set_title('Latency Breakdown by Stage and Memory Type (Side-by-Side)', 
                 fontsize=18, fontweight='bold', pad=20)
    # Use linear scale instead of log scale
    max_total = max([sum(data[config][batch]) for config in configs for batch in batch_sizes])
    ax.set_ylim(0, max_total * 1.1)
    
    # Set x-axis with two-level indexing
    main_positions = []
    for i, config in enumerate(configs):
        group_center = i * 5
        for j, batch in enumerate(batch_sizes):
            batch_center = group_center - 0.5 + j * 0.5
            main_positions.append(batch_center)
    
    ax.set_xticks(main_positions)
    # Remove "batch" and use only numbers
    batch_labels = ['8', '32', '128']
    ax.set_xticklabels(batch_labels * len(configs), fontsize=11)
    
    # Add configuration labels (non-overlapping)
    group_centers = [i * 2.0 for i in range(len(configs))]  # Match the minimal spacing
    for i, (config, pos) in enumerate(zip(configs, group_centers)):
        # Position configuration labels lower to avoid overlap with batch numbers
        ax.text(pos, -max_total * 0.1, config, ha='center', va='top', 
                fontsize=14, fontweight='bold')
    
    # Add lines to separate configuration groups
    for i in range(1, len(configs)):
        sep_pos = (group_centers[i-1] + group_centers[i]) / 2
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, ncol=1)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout with extra bottom margin for two-level labels
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    # Save the figure
    plt.savefig('side_by_side_latency_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig('side_by_side_latency_breakdown.pdf', dpi=300, bbox_inches='tight')
    
    print("Side-by-side figure saved as side_by_side_latency_breakdown.png and .pdf")
    plt.show()
    
    return fig, ax

def create_batch_grouped_breakdown():
    """Create a latency breakdown figure grouped by batch size instead of configuration"""
    
    # Load data from CSV file
    data = load_data_from_csv()
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Colors for different stages
    stage_colors = {
        'draft': '#FF6B6B',     # Red/coral
        'verify': '#4ECDC4',    # Teal/cyan  
        'settle': '#FFD93D'     # Yellow/gold
    }
    
    # Patterns for GPU vs Flash
    gpu_pattern = None      # Solid fill for GPU
    flash_pattern = '///'   # Diagonal lines for Flash
    
    configs = list(data.keys())
    batch_sizes = list(data[configs[0]].keys())
    
    bar_width = 0.25  # Bar width
    x_positions = []
    
    # Calculate positions - group by batch size, then by configuration
    for i, batch in enumerate(batch_sizes):
        batch_center = i * 1.2  # spacing between batch groups
        for j, config in enumerate(configs):
            x_pos = batch_center - bar_width + j * (bar_width + 0.05)
            x_positions.append(x_pos)
    
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
    bottoms = [0] * len(x_positions)
    
    # Track which labels we've added to legend to avoid duplicates
    legend_added = set()
    
    for component_idx, (comp_key, comp_label, color, pattern) in enumerate(components):
        values = []
        
        # Extract values for this component - grouped by batch size first
        for batch in batch_sizes:
            for config in configs:
                values.append(data[config][batch][component_idx])
        
        # Create the bar
        bars = ax.bar(x_positions, values, bar_width, bottom=bottoms,
                     color=color, hatch=pattern, alpha=0.8, 
                     edgecolor='white', linewidth=0.5)
        
        # Add to legend only if not already added
        if comp_label not in legend_added:
            bars[0].set_label(comp_label)
            legend_added.add(comp_label)
        
        # Update bottoms for next layer
        bottoms = [b + v for b, v in zip(bottoms, values)]
    
    # Customize plot
    ax.set_ylabel('Latency (ms)', fontsize=16, fontweight='bold')
    ax.set_title('Latency Breakdown Grouped by Batch Size', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylim(0, max([max(bottoms)] + [0]) * 1.1)
    
    # Set x-axis with two-level indexing (batch size as main groups)
    ax.set_xticks(x_positions)
    # Configuration names as x-tick labels
    config_labels = ['PIM+SD', 'PIM+SD+LB', 'PIM+SD+LB+DV']
    ax.set_xticklabels(config_labels * len(batch_sizes), fontsize=10, rotation=15)
    
    # Add batch size labels below x-axis
    group_centers = [i * 1.2 for i in range(len(batch_sizes))]
    batch_labels = ['8', '32', '128']
    for i, (batch_label, pos) in enumerate(zip(batch_labels, group_centers)):
        ax.text(pos, -max([max(bottoms)] + [0]) * 0.15, f'Batch {batch_label}', 
                ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Add lines to separate batch groups
    for i in range(1, len(batch_sizes)):
        sep_pos = (group_centers[i-1] + group_centers[i]) / 2
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)
    
    # Create custom legend
    legend_elements = []
    
    # Add stage colors with custom labels
    stage_labels = {
        'draft': 'Draft Stage',
        'verify': 'Early Verify Stage', 
        'settle': 'Final Verify Stage'
    }
    for stage, color in stage_colors.items():
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, 
                                           label=stage_labels[stage]))
    
    # Add separator
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='none', edgecolor='none', label=''))
    
    # Add memory type patterns
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.8, 
                                       hatch=gpu_pattern, label='GPU'))
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.8, 
                                       hatch=flash_pattern, label='Flash'))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, 
              frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout with extra bottom margin for two-level labels
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    
    # Save the figure
    plt.savefig('batch_grouped_latency_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig('batch_grouped_latency_breakdown.pdf', dpi=300, bbox_inches='tight')
    
    print("Batch-grouped figure saved as batch_grouped_latency_breakdown.png and .pdf")
    plt.show()
    
    return fig, ax

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
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Colors for different stages
    stage_colors = {
        'draft': '#FF6B6B',     # Red/coral
        'verify': '#4ECDC4',    # Teal/cyan  
        'settle': '#FFD93D'     # Yellow/gold
    }
    
    # Patterns for GPU vs Flash
    gpu_pattern = None      # Solid fill for GPU
    flash_pattern = '///'   # Diagonal lines for Flash
    
    bar_width = 0.25
    x_positions = []
    
    # Calculate positions - group by batch size, then by configuration
    for i, batch in enumerate(batch_sizes):
        batch_center = i * 1.2  # spacing between batch groups
        for j, config in enumerate(configs):
            x_pos = batch_center - bar_width + j * (bar_width + 0.05)
            x_positions.append(x_pos)
    
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
    bottoms = [0] * len(x_positions)
    
    # Track which labels we've added to legend to avoid duplicates
    legend_added = set()
    
    for component_idx, (comp_key, comp_label, color, pattern) in enumerate(components):
        values = []
        
        # Extract normalized values for this component - grouped by batch size first
        for batch in batch_sizes:
            for config in configs:
                values.append(normalized_data[config][batch][component_idx])
        
        # Create the bar
        bars = ax.bar(x_positions, values, bar_width, bottom=bottoms,
                     color=color, hatch=pattern, alpha=0.8, 
                     edgecolor='white', linewidth=0.5)
        
        # Add to legend only if not already added
        if comp_label not in legend_added:
            bars[0].set_label(comp_label)
            legend_added.add(comp_label)
        
        # Update bottoms for next layer
        bottoms = [b + v for b, v in zip(bottoms, values)]
    
    # Customize plot
    ax.set_ylabel('Normalized Latency', fontsize=16, fontweight='bold')
    ax.set_title('Normalized Latency Breakdown (Seq.len 16K)', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylim(0, max([max(bottoms)] + [0]) * 1.1)
    
    # Set x-axis with two-level indexing (batch size as main groups)
    ax.set_xticks(x_positions)
    # Configuration names as x-tick labels
    config_labels = ['PIM+SD', 'PIM+SD+LB', 'PIM+SD+LB+DV']
    ax.set_xticklabels(config_labels * len(batch_sizes), fontsize=10, rotation=15)
    
    # Add batch size labels below x-axis
    group_centers = [i * 1.2 for i in range(len(batch_sizes))]
    batch_labels = ['8', '32', '128']
    for i, (batch_label, pos) in enumerate(zip(batch_labels, group_centers)):
        ax.text(pos, -max([max(bottoms)] + [0]) * 0.15, f'Batch {batch_label}', 
                ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Add lines to separate batch groups
    for i in range(1, len(batch_sizes)):
        sep_pos = (group_centers[i-1] + group_centers[i]) / 2
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)
    
    # Create custom legend
    legend_elements = []
    
    # Add stage colors with custom labels
    stage_labels = {
        'draft': 'Draft Stage',
        'verify': 'Early Verify Stage', 
        'settle': 'Final Verify Stage'
    }
    for stage, color in stage_colors.items():
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, 
                                           label=stage_labels[stage]))
    
    # Add separator
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='none', edgecolor='none', label=''))
    
    # Add memory type patterns
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.8, 
                                       hatch=gpu_pattern, label='GPU'))
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.8, 
                                       hatch=flash_pattern, label='Flash'))
    
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, 
              frameon=True, fancybox=True, shadow=True)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout with extra bottom margin for two-level labels
    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    
    # Save the figure
    plt.savefig('normalized_latency_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig('normalized_latency_breakdown.pdf', dpi=300, bbox_inches='tight')
    
    print("Normalized figure saved as normalized_latency_breakdown.png and .pdf")
    
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
    print("Creating disaggregated latency breakdown figure (stacked with patterns)...")
    create_disaggregated_latency_figure()
    
    print("\nCreating side-by-side latency breakdown figure...")
    create_side_by_side_breakdown()
    
    print("\nCreating batch-grouped latency breakdown figure...")
    create_batch_grouped_breakdown()
    
    print("\nCreating normalized latency breakdown figure...")
    create_normalized_breakdown()
