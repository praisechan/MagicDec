import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ISCA-style plot configuration
def setup_isca_style():
    """Configure matplotlib with ISCA conference style."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['patch.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5

def load_data(csv_file):
    """Load and process data from CSV file."""
    df = pd.read_csv(csv_file)
    
    # Extract the relevant rows
    draft_row = df[df['experiment'].str.contains('report_draft')]
    reject_row = df[df['experiment'].str.contains('budget0.1_taskgov_report_reject') & ~df['experiment'].str.contains('budget_0')]
    budget_025_row = df[df['experiment'].str.contains('budget_0.25_reject')]
    budget_040_row = df[df['experiment'].str.contains('budget_0.40_reject')]
    
    # Get the bin columns (exclude 'experiment' column)
    bin_columns = [col for col in df.columns if col != 'experiment']
    
    # Extract values
    draft_values = draft_row[bin_columns].values[0]
    reject_values = reject_row[bin_columns].values[0]
    budget_025_values = budget_025_row[bin_columns].values[0]
    budget_040_values = budget_040_row[bin_columns].values[0]
    
    # Calculate accepted tokens (draft - reject)
    accepted_values = draft_values - reject_values
    
    # Calculate reject rates (percentages) for each bin
    reject_rate = np.divide(reject_values, draft_values,
                            out=np.zeros_like(reject_values, dtype=float),
                            where=draft_values!=0) * 100
    budget_025_rate = np.divide(budget_025_values, draft_values,
                                out=np.zeros_like(budget_025_values, dtype=float),
                                where=draft_values!=0) * 100
    budget_040_rate = np.divide(budget_040_values, draft_values,
                                out=np.zeros_like(budget_040_values, dtype=float),
                                where=draft_values!=0) * 100
    
    # Create bin centers for x-axis (0.02 width bins)
    bin_centers = np.arange(0.01, 1.0, 0.02)
    
    return {
        'bin_centers': bin_centers,
        'draft_values': draft_values,
        'reject_values': reject_values,
        'budget_025_values': budget_025_values,
        'budget_040_values': budget_040_values,
        'accepted_values': accepted_values,
        'reject_rate': reject_rate,
        'budget_025_rate': budget_025_rate,
        'budget_040_rate': budget_040_rate
    }

def draw_combined_graph(data, output_prefix='histogram_raw_with_budgets'):
    """
    Draw combined histogram and line graph on the same plot.
    
    Args:
        data: Dictionary containing processed data from load_data()
        output_prefix: Prefix for output filenames
    """
    # ISCA-style colors (colorblind-friendly)
    color_accepted = "#696969"
    color_actual = '#e41a1c'
    color_budget25 = '#ffc000'
    color_budget40 = '#ff7f00'
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(7, 2.5))
    
    # Bar chart: draft tokens only
    bar_width = 0.018
    bars_draft = ax1.bar(data['bin_centers'], data['draft_values'], width=0.015, 
                         alpha=0.9, color=color_accepted, edgecolor='black', linewidth=0.8,
                         label='Draft Tokens')
    
    # Configure left y-axis (histogram)
    ax1.set_xlabel('Probability Gap (Top1 - Top2)', fontsize=12)
    ax1.set_ylabel('# Tokens', fontsize=12)
    ax1.tick_params(axis='both', labelsize=12, width=1.5)
    ax1.set_xticks(np.arange(0, 1.2, 0.2))
    ax1.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Create second y-axis for reject rate (percentage)
    ax2 = ax1.twinx()
    
    # Line plots for different budgets with distinct styles
    line_actual = ax2.plot(data['bin_centers'], data['reject_rate'], 
                           color=color_actual, marker='^', linewidth=1.5, markersize=5,
                           label='100% KV', alpha=1.0,
                           markeredgecolor='black', markeredgewidth=0.8, 
                           markevery=2, zorder=5)
    line_025 = ax2.plot(data['bin_centers'], data['budget_025_rate'], 
                        color=color_budget25, marker='s', linewidth=1.5, markersize=4,
                        label='25% KV', alpha=1.0, linestyle='-',
                        markeredgecolor='black', markeredgewidth=0.8,
                        markevery=2, zorder=4)
    line_040 = ax2.plot(data['bin_centers'], data['budget_040_rate'], 
                        color=color_budget40, marker='o', linewidth=1.5, markersize=5,
                        label='40% KV', alpha=1.0, linestyle='-',
                        markeredgecolor='black', markeredgewidth=0.8,
                        markevery=2, zorder=4)
    
    ax2.set_ylabel('Reject Rate (%)', fontsize=12)
    ax2.tick_params(axis='y', labelsize=12, width=1.5)
    max_reject_rate = max(np.max(data['reject_rate']), np.max(data['budget_025_rate']), 
                          np.max(data['budget_040_rate']))
    ax2.set_ylim(0, max_reject_rate * 1.3 if max_reject_rate > 0 else 100)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['top'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    
    # Combine legends
    bars_handles, bars_labels = ax1.get_legend_handles_labels()
    lines_handles = line_actual + line_025 + line_040
    lines_labels = ['100% KV', '25% KV', '40% KV']
    
    # Place legend at upper center
    ax1.legend(bars_handles + lines_handles, bars_labels + lines_labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.23),
               frameon=False, fontsize=12, ncol=4,
               columnspacing=1.0, handlelength=2.0, handletextpad=0.5,
               fancybox=False, edgecolor='black', framealpha=0.95)
    
    plt.tight_layout()
    # Save figures
    plt.savefig(f'/home/juchanlee/MagicDec/figure/confidence/{output_prefix}.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'/home/juchanlee/MagicDec/figure/confidence/{output_prefix}.pdf', 
                bbox_inches='tight')
    print(f"Saved: {output_prefix}.png")
    print(f"Saved: {output_prefix}.pdf")
    
    plt.show()
    plt.close()


def draw_separate_graphs_combined2(data, output_prefix='histogram_separate_combined2'):
    """
    Copy of draw_separate_graphs but the LEFT subplot combines every 2
    original 0.02-width bins into one (approx 0.04 width) and plots the
    combined histogram with reject-rate overlay. The RIGHT subplot is
    kept identical to `draw_separate_graphs`.
    """
    # ISCA-style colors (match existing)
    color_accepted = "#696969"
    color_actual = '#e41a1c'
    color_budget25 = '#ffc000'
    color_budget40 = '#ff7f00'

    # Combine every 2 bins
    n_bins_to_combine = 2
    bin_len = len(data['draft_values'])
    n_combined_bins = bin_len // n_bins_to_combine

    combined_draft = []
    combined_reject = []
    combined_bin_centers = []

    for i in range(n_combined_bins):
        start_idx = i * n_bins_to_combine
        end_idx = start_idx + n_bins_to_combine
        combined_draft.append(np.sum(data['draft_values'][start_idx:end_idx]))
        combined_reject.append(np.sum(data['reject_values'][start_idx:end_idx]))
        # center at middle of the combined bin
        combined_bin_centers.append(np.mean(data['bin_centers'][start_idx:end_idx]))

    combined_draft = np.array(combined_draft)
    combined_reject = np.array(combined_reject)
    combined_bin_centers = np.array(combined_bin_centers)

    # Calculate reject percentage for combined bins
    combined_reject_percentage = np.divide(combined_reject, combined_draft,
                                           out=np.zeros_like(combined_reject, dtype=float),
                                           where=combined_draft!=0) * 100

    # Also combine budgeted reject values (25% and 40%) so the right plot can
    # use the same combined bin centers for lines.
    combined_budget025 = []
    combined_budget040 = []
    for i in range(n_combined_bins):
        start_idx = i * n_bins_to_combine
        end_idx = start_idx + n_bins_to_combine
        combined_budget025.append(np.sum(data['budget_025_values'][start_idx:end_idx]))
        combined_budget040.append(np.sum(data['budget_040_values'][start_idx:end_idx]))

    combined_budget025 = np.array(combined_budget025)
    combined_budget040 = np.array(combined_budget040)

    combined_budget025_percentage = np.divide(combined_budget025, combined_draft,
                                             out=np.zeros_like(combined_budget025, dtype=float),
                                             where=combined_draft!=0) * 100
    combined_budget040_percentage = np.divide(combined_budget040, combined_draft,
                                             out=np.zeros_like(combined_budget040, dtype=float),
                                             where=combined_draft!=0) * 100

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.7))

    # ===== Left plot: Combined-2-bins histogram with reject rate overlay =====
    bars = ax1.bar(combined_bin_centers, combined_draft, width=0.03, alpha=0.9,
                   color=color_accepted, edgecolor='black', linewidth=1.0,
                   label='Draft Tokens')

    ax1.set_xticks(np.arange(0, 1.2, 0.2))
    ax1.set_xlabel('Probability Gap (Top1 - Top2)', fontsize=12)
    ax1.set_ylabel('# Draft Tokens', fontsize=12)
    ax1.tick_params(axis='both', labelsize=12, width=1.5)
    ax1.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)

    # Secondary axis for reject percentage
    ax1_twin = ax1.twinx()
    line_actual = ax1_twin.plot(combined_bin_centers, combined_reject_percentage,
                                color=color_actual, marker='^', linewidth=1.5, markersize=5,
                                label='Reject Rate', alpha=1.0,
                                markeredgecolor='black', markeredgewidth=0.8, zorder=5)

    shared_ylim = (0, 20)
    ax1_twin.set_ylabel('Reject Rate (%)', fontsize=12)
    ax1_twin.tick_params(axis='y', labelsize=12, width=1.5)
    ax1_twin.set_ylim(shared_ylim)
    ax1_twin.spines['right'].set_linewidth(1.5)

    # Legend for left plot (match ISCA combined style)
    bars_handles, bars_labels = ax1.get_legend_handles_labels()
    lines_handles = line_actual
    ax1.legend(bars_handles + lines_handles, bars_labels + ['Reject rate'],
               loc='upper center', bbox_to_anchor=(0.5, 1.22),
               frameon=False, fontsize=12, ncol=2,
               columnspacing=1.0, handlelength=2.0, handletextpad=0.5,
               fancybox=False, edgecolor='black', framealpha=0.95)

    # ===== Right plot: same zoom but use combined bin centers and percentages =====
    zoom_mask = combined_bin_centers <= 0.2
    zoom_centers = combined_bin_centers[zoom_mask]
    zoom_reject_rate = combined_reject_percentage[zoom_mask]
    zoom_budget_025_rate = combined_budget025_percentage[zoom_mask]
    zoom_budget_040_rate = combined_budget040_percentage[zoom_mask]

    line_actual_r = ax2.plot(zoom_centers, zoom_reject_rate, 
                             color=color_actual, marker='^', linewidth=2.0, markersize=4,
                             label='100% KV', alpha=1.0,
                             markeredgecolor='black', markeredgewidth=0.8, 
                             markevery=1)
    line_040_r = ax2.plot(zoom_centers, zoom_budget_040_rate, 
                          color=color_budget40, marker='o', linewidth=2.0, markersize=3,
                          label='40% KV', alpha=1.0, linestyle='-',
                          markeredgecolor='black', markeredgewidth=0.8,
                          markevery=1)
    line_025_r = ax2.plot(zoom_centers, zoom_budget_025_rate, 
                          color=color_budget25, marker='s', linewidth=2.0, markersize=3,
                          label='25% KV', alpha=1.0, linestyle='-',
                          markeredgecolor='black', markeredgewidth=0.8,
                          markevery=1)

    ax2.set_xlabel('Probability Gap (Top1 - Top2)', fontsize=12)
    ax2.tick_params(axis='both', labelsize=12, width=1.5)
    ax2.set_xlim(-0.01, 0.21)
    ax2.set_xticks(np.arange(0, 0.25, 0.05))
    ax2.set_ylim(shared_ylim)
    ax2.grid(True, alpha=0.25, axis='both', linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', fontsize=12, frameon=True, edgecolor='black')

    # Set spine widths
    for ax in [ax1, ax2, ax1_twin]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.tight_layout()
    # Save figures
    plt.savefig(f'/home/juchanlee/MagicDec/figure/confidence/{output_prefix}.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'/home/juchanlee/MagicDec/figure/confidence/{output_prefix}.pdf', 
                bbox_inches='tight')
    print(f"Saved: {output_prefix}.png")
    print(f"Saved: {output_prefix}.pdf")

    plt.show()
    plt.close()

def draw_separate_graphs(data, output_prefix='histogram_separate'):
    """
    Draw separate histogram and line graph side by side.
    
    Args:
        data: Dictionary containing processed data from load_data()
        output_prefix: Prefix for output filenames
    """
    # ISCA-style colors (use same accepted color as plot_histogram_with_reject)
    color_accepted = "#FFC000"
    color_actual = '#e41a1c'
    color_budget25 = '#ffc000'
    color_budget40 = '#ff7f00'
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.7))
    
    # ===== Left plot: Histogram with reject rate overlay =====
    bars_draft = ax1.bar(data['bin_centers'], data['draft_values'], width=0.015, 
                         alpha=0.9, color=color_accepted, edgecolor='black', linewidth=0.8,
                         label='Draft Tokens')
    
    ax1.set_xlabel('Probability Gap (Top1 - Top2)', fontsize=12)
    ax1.set_ylabel('# Draft Tokens', fontsize=12)
    ax1.tick_params(axis='both', labelsize=12, width=1.5)
    ax1.set_xticks(np.arange(0, 1.2, 0.2))
    ax1.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Add reject rate lines on secondary y-axis
    ax1_twin = ax1.twinx()
    line1_actual = ax1_twin.plot(data['bin_centers'], data['reject_rate'], 
                                  color=color_actual, marker='^', linewidth=1.5, markersize=4,
                                  label='Reject rate', alpha=1.0,
                                  markeredgecolor='black', markeredgewidth=0.8, 
                                  markevery=2, zorder=5)
    # line1_025 = ax1_twin.plot(data['bin_centers'], data['budget_025_rate'], 
    #                            color=color_budget25, marker='s', linewidth=1.5, markersize=3,
    #                            label='25% KV', alpha=1.0, linestyle='-',
    #                            markeredgecolor='black', markeredgewidth=0.8,
    #                            markevery=2, zorder=4)
    # line1_040 = ax1_twin.plot(data['bin_centers'], data['budget_040_rate'], 
    #                            color=color_budget40, marker='o', linewidth=1.5, markersize=4,
    #                            label='40% KV', alpha=1.0, linestyle='-',
    #                            markeredgecolor='black', markeredgewidth=0.8,
    #                            markevery=2, zorder=4)
    
    # Calculate shared y-axis scale for reject rate
    # Set fixed maximum to 25%
    shared_ylim = (0, 25)
    
    ax1_twin.set_ylabel('Reject Rate (%)', fontsize=12)
    ax1_twin.tick_params(axis='y', labelsize=12, width=1.5)
    ax1_twin.set_ylim(shared_ylim)
    ax1_twin.spines['right'].set_linewidth(1.5)
    
    # Combine legends for left plot
    bars_handles, bars_labels = ax1.get_legend_handles_labels()
    # lines_handles = line1_actual + line1_025 + line1_040
    lines_handles = line1_actual
    # lines_labels = ['100% KV', '25% KV', '40% KV']
    lines_labels = ['Reject rate']
    ax1.legend(bars_handles + lines_handles, bars_labels + lines_labels,
               loc='upper right', fontsize=10, frameon=True, edgecolor='black')
    
    # ===== Right plot: Line graph only (zoomed to 0.0-0.3) =====
    # Filter data for zoom range
    zoom_mask = data['bin_centers'] <= 0.2
    zoom_centers = data['bin_centers'][zoom_mask]
    zoom_reject_rate = data['reject_rate'][zoom_mask]
    zoom_budget_025_rate = data['budget_025_rate'][zoom_mask]
    zoom_budget_040_rate = data['budget_040_rate'][zoom_mask]
    
    line_actual = ax2.plot(zoom_centers, zoom_reject_rate, 
                           color=color_actual, marker='^', linewidth=2.0, markersize=4,
                           label='100% KV', alpha=1.0,
                           markeredgecolor='black', markeredgewidth=0.8, 
                           markevery=1)
    line_040 = ax2.plot(zoom_centers, zoom_budget_040_rate, 
                        color=color_budget40, marker='o', linewidth=2.0, markersize=3,
                        label='40% KV', alpha=1.0, linestyle='-',
                        markeredgecolor='black', markeredgewidth=0.8,
                        markevery=1)
    line_025 = ax2.plot(zoom_centers, zoom_budget_025_rate, 
                        color=color_budget25, marker='s', linewidth=2.0, markersize=3,
                        label='25% KV', alpha=1.0, linestyle='-',
                        markeredgecolor='black', markeredgewidth=0.8,
                        markevery=1)
    
    ax2.set_xlabel('Probability Gap (Top1 - Top2)', fontsize=12)
    # Remove y-axis label for right plot to avoid redundancy
    # ax2.set_ylabel('Reject Rate (%)', fontsize=12)
    ax2.tick_params(axis='both', labelsize=12, width=1.5)
    ax2.set_xlim(-0.01, 0.21)  # Zoom to 0.0-0.2
    ax2.set_xticks(np.arange(0, 0.25, 0.05))
    # Use the same y-axis scale as the left plot
    ax2.set_ylim(shared_ylim)
    ax2.grid(True, alpha=0.25, axis='both', linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', fontsize=12, frameon=True, edgecolor='black')
    # ax2.set_title('(b) Reject Rate (Zoom: 0.0-0.3)', fontsize=13, fontweight='bold', pad=10)
    
    # Set spine widths
    for ax in [ax1, ax2, ax1_twin]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    plt.tight_layout()
    # Save figures
    plt.savefig(f'/home/juchanlee/MagicDec/figure/confidence/{output_prefix}.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'/home/juchanlee/MagicDec/figure/confidence/{output_prefix}.pdf', 
                bbox_inches='tight')
    print(f"Saved: {output_prefix}.png")
    print(f"Saved: {output_prefix}.pdf")
    
    plt.show()
    plt.close()


def draw_separate_graphs_with_reject_style(data, output_prefix='histogram_separate_reject_style'):
    """
    Variant of draw_separate_graphs where the LEFT subplot is replaced by
    the combined-bin histogram + reject-rate overlay style from
    `plot_histogram_with_reject.py` (combined bins of width 0.1).
    """
    # ISCA-style colors
    color_accepted = "#696969"
    color_actual = '#e41a1c'
    color_budget25 = '#ffc000'
    color_budget40 = '#ff7f00'

    # Combine every 5 bins (0.02 * 5 = 0.1) for left histogram
    n_bins_to_combine = 5
    bin_len = len(data['draft_values'])
    n_combined_bins = bin_len // n_bins_to_combine

    combined_draft = []
    combined_reject = []
    combined_bin_centers = []

    for i in range(n_combined_bins):
        start_idx = i * n_bins_to_combine
        end_idx = start_idx + n_bins_to_combine
        combined_draft.append(np.sum(data['draft_values'][start_idx:end_idx]))
        combined_reject.append(np.sum(data['reject_values'][start_idx:end_idx]))
        combined_bin_centers.append(i * 0.1 + 0.05)

    combined_draft = np.array(combined_draft)
    combined_reject = np.array(combined_reject)
    combined_bin_centers = np.array(combined_bin_centers)

    # Calculate reject percentage for combined bins
    combined_reject_percentage = np.divide(combined_reject, combined_draft,
                                           out=np.zeros_like(combined_reject, dtype=float),
                                           where=combined_draft!=0) * 100

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.7))

    # ===== Left plot: Combined histogram with reject rate overlay (from plot_histogram_with_reject) =====
    bars2 = ax1.bar(combined_bin_centers, combined_draft, width=0.06, alpha=0.9,
                    color=color_accepted, edgecolor='black', linewidth=1.2, label='Draft Tokens')

    ax1.set_xticks(np.arange(0, 1.2, 0.2))
    ax1.set_xlabel('Probability Gap (Top1 - Top2)', fontsize=12)
    ax1.set_ylabel('# Draft Tokens', fontsize=12)
    ax1.tick_params(axis='both', labelsize=12, width=1.5)
    ax1.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)

    # Secondary axis for reject percentage
    ax1_twin = ax1.twinx()
    line1_actual = ax1_twin.plot(combined_bin_centers, combined_reject_percentage,
                                  color=color_actual, marker='^', linewidth=1.5, markersize=5,
                                  label='Reject Rate', alpha=1.0,
                                  markeredgecolor='black', markeredgewidth=0.8, zorder=5)

    # Fixed y-axis for reject rate (match plot_histogram_with_reject behavior)
    shared_ylim = (0, 25)
    ax1_twin.set_ylabel('Reject Rate (%)', fontsize=12)
    ax1_twin.tick_params(axis='y', labelsize=12, width=1.5)
    ax1_twin.set_ylim(shared_ylim)
    ax1_twin.spines['right'].set_linewidth(1.5)

    # Combine legends for left plot (follow plot_histogram_with_reject legend style)
    bars_handles, bars_labels = ax1.get_legend_handles_labels()
    lines_handles = line1_actual
    lines_labels = ['Reject rate']
    ax1.legend(bars_handles + lines_handles, bars_labels + lines_labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.18),
               frameon=False, fontsize=12, ncol=2,
               columnspacing=1.0, handlelength=2.0, handletextpad=0.5,
               fancybox=False, edgecolor='black', framealpha=0.95)

    # ===== Right plot: Line graph only (zoomed to 0.0-0.3) - borrow from draw_separate_graphs =====
    zoom_mask = data['bin_centers'] <= 0.2
    zoom_centers = data['bin_centers'][zoom_mask]
    zoom_reject_rate = data['reject_rate'][zoom_mask]
    zoom_budget_025_rate = data['budget_025_rate'][zoom_mask]
    zoom_budget_040_rate = data['budget_040_rate'][zoom_mask]

    line_actual = ax2.plot(zoom_centers, zoom_reject_rate, 
                           color=color_actual, marker='^', linewidth=2.0, markersize=4,
                           label='100% KV', alpha=1.0,
                           markeredgecolor='black', markeredgewidth=0.8, 
                           markevery=1)
    line_040 = ax2.plot(zoom_centers, zoom_budget_040_rate, 
                        color=color_budget40, marker='o', linewidth=2.0, markersize=3,
                        label='40% KV', alpha=1.0, linestyle='-',
                        markeredgecolor='black', markeredgewidth=0.8,
                        markevery=1)
    line_025 = ax2.plot(zoom_centers, zoom_budget_025_rate, 
                        color=color_budget25, marker='s', linewidth=2.0, markersize=3,
                        label='25% KV', alpha=1.0, linestyle='-',
                        markeredgecolor='black', markeredgewidth=0.8,
                        markevery=1)

    ax2.set_xlabel('Probability Gap (Top1 - Top2)', fontsize=12)
    ax2.tick_params(axis='both', labelsize=12, width=1.5)
    ax2.set_xlim(-0.01, 0.21)
    ax2.set_xticks(np.arange(0, 0.25, 0.05))
    ax2.set_ylim(shared_ylim)
    ax2.grid(True, alpha=0.25, axis='both', linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', fontsize=12, frameon=True, edgecolor='black')

    # Set spine widths
    for ax in [ax1, ax2, ax1_twin]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.tight_layout()
    # Save figures
    plt.savefig(f'/home/juchanlee/MagicDec/figure/confidence/{output_prefix}.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'/home/juchanlee/MagicDec/figure/confidence/{output_prefix}.pdf', 
                bbox_inches='tight')
    print(f"Saved: {output_prefix}.png")
    print(f"Saved: {output_prefix}.pdf")

    plt.show()
    plt.close()

def print_statistics(data):
    """Print summary statistics."""
    print("\n=== Statistics ===")
    print(f"Total draft tokens: {np.sum(data['draft_values']):.0f}")
    print(f"Total accepted tokens: {np.sum(data['accepted_values']):.0f}")
    print(f"Total rejected tokens (baseline): {np.sum(data['reject_values']):.0f}")
    print(f"Overall reject rate (baseline): {np.sum(data['reject_values']) / np.sum(data['draft_values']) * 100:.2f}%")
    print(f"\nTotal rejected tokens (budget 0.25): {np.sum(data['budget_025_values']):.0f}")
    print(f"Reject rate (budget 0.25): {np.sum(data['budget_025_values']) / np.sum(data['draft_values']) * 100:.2f}%")
    print(f"\nTotal rejected tokens (budget 0.40): {np.sum(data['budget_040_values']):.0f}")
    print(f"Reject rate (budget 0.40): {np.sum(data['budget_040_values']) / np.sum(data['draft_values']) * 100:.2f}%")
    print(f"\nMax reject rate per bin (baseline): {np.max(data['reject_rate']):.2f}% at bin center {data['bin_centers'][np.argmax(data['reject_rate'])]:.2f}")
    print(f"Max reject rate per bin (budget 0.25): {np.max(data['budget_025_rate']):.2f}% at bin center {data['bin_centers'][np.argmax(data['budget_025_rate'])]:.2f}")
    print(f"Max reject rate per bin (budget 0.40): {np.max(data['budget_040_rate']):.2f}% at bin center {data['bin_centers'][np.argmax(data['budget_040_rate'])]:.2f}")
    print(f"\nVerification: Draft = Accepted + Rejected?")
    print(f"  {np.sum(data['draft_values']):.0f} = {np.sum(data['accepted_values']):.0f} + {np.sum(data['reject_values']):.0f}")
    print(f"  Match: {np.isclose(np.sum(data['draft_values']), np.sum(data['accepted_values']) + np.sum(data['reject_values']))}")

# Main execution
if __name__ == "__main__":
    # Setup style
    setup_isca_style()
    
    csv_file = '/home/juchanlee/MagicDec/figure/confidence/data/run2step_Meta-Llama-3.1-8B_pg19_histogram_data_32768_new.csv'
    # csv_file = '/home/juchanlee/MagicDec/figure/confidence/data/run2step_Meta-Llama-3.1-8B_pg19_histogram_data_32768_multibudget.csv'
    
    # Load data
    data = load_data(csv_file)
    
    # Draw combined graph (original style)
    print("\n=== Drawing Combined Graph ===")
    draw_combined_graph(data, output_prefix='histogram_raw_with_budgets')
    
    # Draw separate graphs (histogram and line graph side by side)
    print("\n=== Drawing Separate Graphs ===")
    draw_separate_graphs(data, output_prefix='histogram_separate')
    
    draw_separate_graphs_with_reject_style(data, output_prefix='histogram_separate_reject_style')
    
    draw_separate_graphs_combined2(data, output_prefix='histogram_separate_combined2')
    
    # Print statistics
    print_statistics(data)
