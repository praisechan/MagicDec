import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

def create_latency_breakdown_chart(csv_filepath):
    """
    Generates a stacked bar chart for latency breakdown from a given CSV file.

    This function is designed to be robust against malformed CSV headers by
    skipping them and defining the data structure manually.
    """
    try:
        # --- 1. Robust Data Loading and Processing ---

        # Define the correct column names manually, including the final 'Total' column.
        column_names = [
            'Case', 'Batch Size', 
            'draft_GPU', 'draft_Flash', 
            'verify1_GPU', 'verify1_Flash', 
            'settle_GPU', 'settle_Flash', 
            'Total'
        ]
        
        # Read the CSV, skipping the two malformed header rows and applying our own names.
        df = pd.read_csv(csv_filepath, skiprows=2, header=None, names=column_names)

        # Drop the unneeded 'Total' column.
        df.drop(columns='Total', inplace=True)

        # Forward-fill the 'Case' column to populate the empty cells.
        df['Case'].ffill(inplace=True)

        # Remove any empty rows that might exist between data groups.
        df.dropna(subset=df.columns[2:], how='all', inplace=True)

        # Standardize the 'Batch Size' column.
        df['Batch Size'] = df['Batch Size'].str.strip().str.replace(' ', '')
        
        # --- 2. Data Preparation for Plotting ---
        
        # Define the desired order for cases and batches for consistent plotting.
        batch_order = ['batch8', 'batch32', 'batch128']
        df['Batch Size'] = pd.Categorical(df['Batch Size'], categories=batch_order, ordered=True)
        
        # Sort the DataFrame.
        df.sort_values(by=['Batch Size', 'Case'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        cases = df['Case'].unique()
        n_cases = len(cases)
        n_batches = len(batch_order)
        
        # Define the visual properties for the chart.
        colors = {'draft': '#1f77b4', 'verify1': '#ff7f0e', 'settle': '#2ca02c'}
        hatches = {'GPU': '', 'Flash': '///'} # Use empty string for solid fill
        
        # Get the columns that contain the latency data to be stacked.
        latency_cols = df.columns[2:]

        # --- 3. Plotting the Chart ---

        fig, ax = plt.subplots(figsize=(18, 10))
        bar_width = 0.7
        indices = np.arange(len(df))
        bottom = np.zeros(len(df))

        # Loop through each latency component to create the stacks.
        for col_name in latency_cols:
            stage, latency_type = col_name.split('_')
            values = df[col_name].astype(float)
            
            ax.bar(
                indices, 
                values, 
                bar_width, 
                bottom=bottom,
                color=colors[stage],
                hatch=hatches[latency_type],
                edgecolor='white',
                linewidth=0.5
            )
            bottom += values

        # --- 4. Customizing the X-axis for Two Levels ---

        ax.set_xticks(indices)
        ax.set_xticklabels(df['Case'], rotation=45, ha='right', fontsize=12)
        ax.tick_params(axis='x', which='major', length=0)
        
        # Add visual separators for batch size groups.
        for i in range(n_batches - 1):
            separator_pos = (i + 1) * n_cases - 0.5
            ax.axvline(x=separator_pos, color='gray', linestyle='--', linewidth=1)

        # Add the higher-level 'Batch Size' labels.
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15) 
        for i, batch in enumerate(batch_order):
            group_midpoint = i * n_cases + (n_cases / 2.0) - 0.5
            ax.text(
                group_midpoint, ax.get_ylim()[1] * 0.99, f'Batch Size: {batch[5:]}',
                ha='center', va='bottom', fontsize=14, fontweight='bold'
            )

        # --- 5. Final Touches (Labels, Title, Legend) ---
        
        ax.set_ylabel('Latency (ms)', fontsize=14)
        ax.set_title('Simulation Latency Breakdown by Case and Batch Size', fontsize=18, pad=40)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Create a custom legend.
        legend_elements = [
            Patch(facecolor=colors['draft'], label='Draft Stage'),
            Patch(facecolor=colors['verify1'], label='Verify Stage'),
            Patch(facecolor=colors['settle'], label='Settle Stage'),
            Patch(facecolor='grey', label='GPU Latency'),
            Patch(facecolor='grey', hatch='///', label='Flash Latency')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12, bbox_to_anchor=(1.01, 1))
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig('latency_breakdown_chart.png', dpi=300, bbox_inches='tight')
        
        print("Chart has been generated and saved as 'latency_breakdown_chart.png'")

    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    csv_file = 'simulation_latency_breakdown_qwen14b_16K.CSV'
    create_latency_breakdown_chart(csv_file)

