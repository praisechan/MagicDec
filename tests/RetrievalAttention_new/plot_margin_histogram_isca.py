import argparse
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def parse_args():
    parser = argparse.ArgumentParser(description="Plot confidence-margin histogram + reject-rate line (ISCA style).")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def load_histogram_csv(path):
    bin_left = []
    bin_right = []
    bin_center = []
    drafted_count = []
    reject_rate = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bin_left.append(float(row["bin_left"]))
            bin_right.append(float(row["bin_right"]))
            bin_center.append(float(row["bin_center"]))
            drafted_count.append(float(row["drafted_token_count"]))
            reject_rate.append(float(row["reject_rate"]) * 100.0)

    return bin_left, bin_right, bin_center, drafted_count, reject_rate


def main():
    args = parse_args()
    input_csv = os.path.abspath(args.input_csv)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"input_csv not found: {input_csv}")

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(input_csv))[0]
    output_png = os.path.join(output_dir, f"{stem}_isca.png")
    output_pdf = os.path.join(output_dir, f"{stem}_isca.pdf")

    left, right, centers, counts, reject_rate_pct = load_histogram_csv(input_csv)
    widths = [r - l for l, r in zip(left, right)]

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

    fig, ax_count = plt.subplots(figsize=(7.0, 4.0))

    ax_count.bar(
        centers,
        counts,
        width=widths,
        color="#B0B0B0",
        edgecolor="#6E6E6E",
        linewidth=0.7,
        align="center",
        label="Draft Token Count",
        zorder=2,
    )
    ax_count.set_xlim(0.0, 1.0)
    ax_count.set_xlabel("Probability Gap (Top1 - Top2)")
    ax_count.set_ylabel("# Draft Tokens")
    ax_count.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55, zorder=0)

    ax_reject = ax_count.twinx()
    ax_reject.plot(
        centers,
        reject_rate_pct,
        color="#C62828",
        marker="^",
        markersize=4.5,
        linewidth=1.6,
        label="Reject Rate",
        zorder=3,
    )
    ax_reject.set_ylabel("Reject Rate (%)", color="#C62828")
    ax_reject.tick_params(axis="y", colors="#C62828")
    ax_reject.yaxis.set_major_formatter(PercentFormatter(xmax=100.0, decimals=0))

    handles_left, labels_left = ax_count.get_legend_handles_labels()
    handles_right, labels_right = ax_reject.get_legend_handles_labels()
    ax_count.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_png, dpi=350, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=350, bbox_inches="tight")
    plt.close(fig)

    print(f"input_csv: {input_csv}")
    print(f"output_png: {output_png}")
    print(f"output_pdf: {output_pdf}")


if __name__ == "__main__":
    main()
