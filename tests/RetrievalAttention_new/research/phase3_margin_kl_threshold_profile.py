import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MARGIN_BINS = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 1.01]
KL_THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.03]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile accepted-prefix token distribution and reject rate over early-margin bins at multiple KL thresholds."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_png", type=str, default="")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--output_prefix", type=str, default="phase3_margin_kl_threshold_profile")
    return parser.parse_args()


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_profile(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[
        (df["is_bonus_position"] == 0)
        & (df["pre_final_features_available"] == 1)
        & (df["source_kind"] == "accepted_prefix")
    ].copy()
    df["margin_bin"] = pd.cut(
        df["early_margin"],
        bins=MARGIN_BINS,
        include_lowest=True,
        right=False,
    )
    return df


def margin_bin_labels() -> List[str]:
    labels = []
    for left, right in zip(MARGIN_BINS[:-1], MARGIN_BINS[1:]):
        labels.append(f"[{left:.3f}, {right:.3f})")
    return labels


def compute_threshold_profile(df: pd.DataFrame) -> pd.DataFrame:
    ordered_bins = pd.IntervalIndex.from_breaks(MARGIN_BINS, closed="left")
    rows: List[Dict[str, object]] = []

    for threshold in KL_THRESHOLDS:
        subset = df[df["kl_early_draft"] >= threshold].copy()
        total_tokens = len(subset)
        grouped = (
            subset.groupby("margin_bin", observed=False)["is_rejected_position"]
            .agg(["count", "sum", "mean"])
            .reindex(ordered_bins, fill_value=0)
            .reset_index()
        )
        if "margin_bin" not in grouped.columns and "index" in grouped.columns:
            grouped = grouped.rename(columns={"index": "margin_bin"})
        for row in grouped.to_dict("records"):
            interval = row["margin_bin"]
            token_count = int(row["count"])
            reject_count = int(row["sum"])
            reject_rate = float(row["mean"]) if token_count > 0 else 0.0
            token_share = (token_count / total_tokens) if total_tokens > 0 else 0.0
            rows.append(
                {
                    "kl_threshold": threshold,
                    "kl_label": f"KL >= {threshold:.3f}",
                    "margin_bin_left": float(interval.left),
                    "margin_bin_right": float(interval.right),
                    "margin_bin_label": f"[{interval.left:.3f}, {interval.right:.3f})",
                    "token_count": token_count,
                    "token_share": token_share,
                    "rejected_token_count": reject_count,
                    "reject_rate": reject_rate,
                }
            )

    return pd.DataFrame(rows)


def plot_profile(summary: pd.DataFrame, title: str, output_png: str):
    labels = margin_bin_labels()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(KL_THRESHOLDS)))

    for color, threshold in zip(colors, KL_THRESHOLDS):
        subset = summary[summary["kl_threshold"] == threshold]
        axes[0].plot(
            x,
            subset["token_share"].to_numpy() * 100.0,
            marker="o",
            linewidth=2,
            color=color,
            label=f"KL >= {threshold:.3f}",
        )
        axes[1].plot(
            x,
            subset["reject_rate"].to_numpy() * 100.0,
            marker="o",
            linewidth=2,
            color=color,
            label=f"KL >= {threshold:.3f}",
        )

    axes[0].set_title("Token Distribution by Margin Bin")
    axes[0].set_ylabel("Token share (%)")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].set_title("Reject Rate by Margin Bin")
    axes[1].set_ylabel("Reject rate (%)")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlabel("early_margin bin")
        ax.legend(loc="best", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = os.path.dirname(args.input_csv)
    output_png = args.output_png or os.path.join(input_dir, f"{args.output_prefix}.png")
    output_csv = args.output_csv or os.path.join(input_dir, f"{args.output_prefix}.csv")

    ensure_parent(output_png)
    ensure_parent(output_csv)

    df = load_profile(args.input_csv)
    summary = compute_threshold_profile(df)
    dataset_title = args.title or os.path.basename(args.input_csv)

    summary.to_csv(output_csv, index=False)
    plot_profile(summary, dataset_title, output_png)

    print(f"Wrote summary CSV to {output_csv}")
    print(f"Wrote figure PNG to {output_png}")


if __name__ == "__main__":
    main()
