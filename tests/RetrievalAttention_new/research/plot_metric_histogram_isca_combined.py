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
        description="Plot token distribution and reject rate from one or more metric histogram CSVs."
    )
    parser.add_argument("--csv_files", nargs="+", required=True)
    parser.add_argument("--legend_names", nargs="+", default=None)
    parser.add_argument("--bin_width", type=float, required=True)
    parser.add_argument("--x_label", type=str, default="Metric Value")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_stem", type=str, default="metric_histogram_isca")
    parser.add_argument("--x_min", type=float, default=None)
    parser.add_argument("--x_max", type=float, default=None)
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

    source_bin_width = rows[0]["bin_right"] - rows[0]["bin_left"]
    merge_factor_float = target_bin_width / source_bin_width
    merge_factor = round(merge_factor_float)
    if not math.isclose(merge_factor_float, merge_factor, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"bin_width {target_bin_width} must be an integer multiple of the source CSV bin width {source_bin_width} "
            f"in {csv_path}."
        )

    merged_centers = []
    merged_counts = []
    merged_reject_rates = []
    merged_lefts = []
    merged_rights = []

    for start_idx in range(0, len(rows), merge_factor):
        chunk = rows[start_idx : start_idx + merge_factor]
        merged_left = chunk[0]["bin_left"]
        merged_right = chunk[-1]["bin_right"]
        drafted_count = sum(row["drafted_token_count"] for row in chunk)
        rejected_count = sum(row["rejected_token_count"] for row in chunk)
        reject_rate_pct = (rejected_count / drafted_count) * 100.0 if drafted_count > 0.0 else 0.0

        merged_lefts.append(merged_left)
        merged_rights.append(merged_right)
        merged_centers.append((merged_left + merged_right) / 2.0)
        merged_counts.append(drafted_count)
        merged_reject_rates.append(reject_rate_pct)

    return merged_lefts, merged_rights, merged_centers, merged_counts, merged_reject_rates


def build_distribution_pct(counts):
    total = sum(counts)
    if total <= 0.0:
        return [0.0 for _ in counts]
    return [(count / total) * 100.0 for count in counts]


def plot_combined_figure(series_list, legend_names, output_png, output_pdf, x_label, x_min, x_max):
    fig, (ax_dist, ax_reject) = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=False)
    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    legend_handles = []
    for idx, (series, legend_name) in enumerate(zip(series_list, legend_names)):
        color = cmap(idx % cmap.N)
        marker = markers[idx % len(markers)]

        (dist_line,) = ax_dist.plot(
            series["centers"],
            series["distribution_pct"],
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
        legend_handles.append(dist_line)

    for ax in (ax_dist, ax_reject):
        ax.set_xlim(x_min, x_max)
        ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.55)
        ax.set_xlabel(x_label)

    ax_dist.set_title("Token Distribution")
    ax_dist.set_ylabel("Token Distribution (%)")
    ax_dist.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))

    ax_reject.set_title("Reject Rate")
    ax_reject.set_ylabel("Reject Rate (%)")
    ax_reject.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))

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


def main():
    args = parse_args()
    csv_files = [os.path.abspath(path) for path in parse_list_argument(args.csv_files)]
    legend_names = (
        parse_list_argument(args.legend_names)
        if args.legend_names is not None
        else [os.path.splitext(os.path.basename(path))[0] for path in csv_files]
    )
    if len(legend_names) != len(csv_files):
        raise ValueError("legend_names must match csv_files length.")

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(csv_files[0])
    os.makedirs(output_dir, exist_ok=True)

    output_png = os.path.join(output_dir, f"{args.output_stem}.png")
    output_pdf = os.path.join(output_dir, f"{args.output_stem}.pdf")

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

    series_list = []
    global_left = None
    global_right = None
    for csv_path in csv_files:
        rows = load_histogram_csv(csv_path)
        lefts, rights, centers, counts, reject_rate_pct = merge_histogram_rows(rows, args.bin_width, csv_path)
        series_list.append(
            {
                "centers": centers,
                "distribution_pct": build_distribution_pct(counts),
                "reject_rate_pct": reject_rate_pct,
            }
        )
        local_left = min(lefts)
        local_right = max(rights)
        global_left = local_left if global_left is None else min(global_left, local_left)
        global_right = local_right if global_right is None else max(global_right, local_right)

    plot_combined_figure(
        series_list,
        legend_names,
        output_png,
        output_pdf,
        args.x_label,
        args.x_min if args.x_min is not None else global_left,
        args.x_max if args.x_max is not None else global_right,
    )

    print("input_csv_files:")
    for csv_path in csv_files:
        print(f"  - {csv_path}")
    print(f"output_png: {output_png}")
    print(f"output_pdf: {output_pdf}")


if __name__ == "__main__":
    main()
