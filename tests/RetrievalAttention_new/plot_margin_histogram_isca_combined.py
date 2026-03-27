"""
Sample usage:
python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/plot_margin_histogram_isca_combined.py \
  --csv_files csv1.csv,csv2.csv,csv3.csv \
  --legend_names name1,name2,name3 \
  --bin_width 0.1 \
  --output_stem combined_margin_plot
"""
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_list_argument(values):
    items = []
    for value in values:
        for part in value.split(","):
            stripped = part.strip()
            if stripped:
                items.append(stripped)
    return items


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot draft token counts and reject rates from multiple margin-profile CSVs in one combined ISCA-style figure."
    )
    parser.add_argument(
        "--csv_files",
        "--csv-files",
        "--csv_file",
        "--csv-file",
        "--input_csvs",
        "--csvs",
        nargs="+",
        required=True,
        help="CSV paths, provided either as repeated values or comma-separated values.",
    )
    parser.add_argument(
        "--legend_names",
        "--legend-names",
        "--legend_name",
        "--legend-name",
        nargs="+",
        default=None,
        help="Legend labels, provided either as repeated values or comma-separated values.",
    )
    parser.add_argument(
        "--bin_width",
        "--bin-width",
        type=float,
        default=0.02,
        help="Target bin width for the combined figure. It must be an integer multiple of the source CSV bin width.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_stem", type=str, default="combined_margin_histogram_isca")
    return parser.parse_args()


def load_histogram_csv(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "bin_left": float(row["bin_left"]),
                    "bin_right": float(row["bin_right"]),
                    "drafted_token_count": float(row["drafted_token_count"]),
                    "rejected_token_count": float(row["rejected_token_count"]),
                }
            )

    return rows


def merge_histogram_rows(rows, target_bin_width, csv_path):
    if not rows:
        raise ValueError(f"No histogram rows found in CSV: {csv_path}")
    if target_bin_width <= 0.0:
        raise ValueError(f"bin_width must be positive, got {target_bin_width}.")

    source_bin_width = rows[0]["bin_right"] - rows[0]["bin_left"]
    if source_bin_width <= 0.0:
        raise ValueError(f"Invalid source bin width {source_bin_width} in CSV: {csv_path}")

    for row in rows:
        row_bin_width = row["bin_right"] - row["bin_left"]
        if not math.isclose(row_bin_width, source_bin_width, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"CSV contains inconsistent bin widths in {csv_path}. "
                f"Expected {source_bin_width}, got {row_bin_width}."
            )

    if target_bin_width + 1e-9 < source_bin_width:
        raise ValueError(
            f"bin_width {target_bin_width} is smaller than the source CSV bin width {source_bin_width} in {csv_path}."
        )

    merge_factor_float = target_bin_width / source_bin_width
    merge_factor = round(merge_factor_float)
    if not math.isclose(merge_factor_float, merge_factor, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"bin_width {target_bin_width} must be an integer multiple of the source CSV bin width {source_bin_width} "
            f"in {csv_path}."
        )

    merged_centers = []
    merged_drafted_counts = []
    merged_reject_rates = []

    for start_idx in range(0, len(rows), merge_factor):
        chunk = rows[start_idx : start_idx + merge_factor]
        merged_left = chunk[0]["bin_left"]
        merged_right = chunk[-1]["bin_right"]
        drafted_count = sum(row["drafted_token_count"] for row in chunk)
        rejected_count = sum(row["rejected_token_count"] for row in chunk)
        reject_rate_pct = (rejected_count / drafted_count) * 100.0 if drafted_count > 0.0 else 0.0

        merged_centers.append((merged_left + merged_right) / 2.0)
        merged_drafted_counts.append(drafted_count)
        merged_reject_rates.append(reject_rate_pct)

    return merged_centers, merged_drafted_counts, merged_reject_rates


def build_draft_distribution_pct(drafted_counts):
    total_count = sum(drafted_counts)
    if total_count <= 0.0:
        return [0.0 for _ in drafted_counts]
    return [(count / total_count) * 100.0 for count in drafted_counts]


def plot_combined_figure(series_list, legend_names, output_png, output_pdf, left_title, left_ylabel, left_as_percent):
    fig, (ax_count, ax_reject) = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=False)
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    legend_handles = []
    for idx, (series, legend_name) in enumerate(zip(series_list, legend_names)):
        color = cmap(idx % cmap.N)
        marker = markers[idx % len(markers)]

        (count_line,) = ax_count.plot(
            series["centers"],
            series["left_values"],
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.8,
            label=legend_name,
        )
        ax_reject.plot(
            series["centers"],
            series["reject_rate_pct"],
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.8,
            label=legend_name,
        )
        legend_handles.append(count_line)

    ax_count.set_title(left_title)
    ax_count.set_xlabel("Probability Gap (Top1 - Top2)")
    ax_count.set_ylabel(left_ylabel)
    ax_count.set_xlim(0.0, 1.0)
    if left_as_percent:
        ax_count.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax_count.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.55)

    ax_reject.set_title("Reject Rate")
    ax_reject.set_xlabel("Probability Gap (Top1 - Top2)")
    ax_reject.set_ylabel("Reject Rate (%)")
    ax_reject.set_xlim(0.0, 1.0)
    ax_reject.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))
    ax_reject.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.55)

    fig.legend(
        legend_handles,
        legend_names,
        loc="upper center",
        ncol=min(len(legend_names), 4),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(output_png, dpi=350, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=350, bbox_inches="tight")
    plt.close(fig)


def resolve_input_paths(csv_values):
    csv_files = [os.path.abspath(path) for path in parse_list_argument(csv_values)]
    if not csv_files:
        raise ValueError("At least one CSV file must be provided.")

    missing_files = [path for path in csv_files if not os.path.exists(path)]
    if missing_files:
        missing = "\n".join(missing_files)
        raise FileNotFoundError(f"CSV file(s) not found:\n{missing}")

    return csv_files


def resolve_legend_names(csv_files, legend_values):
    if legend_values is None:
        return [os.path.splitext(os.path.basename(path))[0] for path in csv_files]

    legend_names = parse_list_argument(legend_values)
    if len(legend_names) != len(csv_files):
        raise ValueError(
            f"Expected {len(csv_files)} legend names for {len(csv_files)} CSV files, got {len(legend_names)}."
        )

    return legend_names


def main():
    args = parse_args()
    csv_files = resolve_input_paths(args.csv_files)
    legend_names = resolve_legend_names(csv_files, args.legend_names)

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(csv_files[0])
    os.makedirs(output_dir, exist_ok=True)

    output_png = os.path.join(output_dir, f"{args.output_stem}.png")
    output_pdf = os.path.join(output_dir, f"{args.output_stem}.pdf")
    output_dist_png = os.path.join(output_dir, f"{args.output_stem}_distribution.png")
    output_dist_pdf = os.path.join(output_dir, f"{args.output_stem}_distribution.pdf")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    count_series_list = []
    distribution_series_list = []
    for csv_path in csv_files:
        rows = load_histogram_csv(csv_path)
        centers, counts, reject_rate_pct = merge_histogram_rows(rows, args.bin_width, csv_path)
        count_series_list.append(
            {
                "centers": centers,
                "left_values": counts,
                "reject_rate_pct": reject_rate_pct,
            }
        )
        distribution_series_list.append(
            {
                "centers": centers,
                "left_values": build_draft_distribution_pct(counts),
                "reject_rate_pct": reject_rate_pct,
            }
        )

    plot_combined_figure(
        count_series_list,
        legend_names,
        output_png,
        output_pdf,
        left_title="Draft Token Count",
        left_ylabel="# Draft Tokens",
        left_as_percent=False,
    )
    plot_combined_figure(
        distribution_series_list,
        legend_names,
        output_dist_png,
        output_dist_pdf,
        left_title="Draft Token Distribution",
        left_ylabel="Draft Token Distribution (%)",
        left_as_percent=True,
    )

    print("input_csv_files:")
    for csv_path in csv_files:
        print(f"  - {csv_path}")
    print(f"bin_width: {args.bin_width}")
    print(f"output_png: {output_png}")
    print(f"output_pdf: {output_pdf}")
    print(f"output_distribution_png: {output_dist_png}")
    print(f"output_distribution_pdf: {output_dist_pdf}")


if __name__ == "__main__":
    main()
