import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_MARGIN_THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.20]
DEFAULT_KL_THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.03]


def parse_float_list(value: str, default: List[float]) -> List[float]:
    if not value:
        return list(default)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile block-level rejection-indicator precision/recall over early-margin thresholds "
            "for several KL(early_verify || draft) thresholds."
        )
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_png", type=str, default="")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument(
        "--source_kind",
        type=str,
        default="accepted_prefix",
        choices=["accepted_prefix", "early_bonus", "all"],
        help="Which source rows are eligible to trigger the indicator.",
    )
    parser.add_argument(
        "--margin_thresholds",
        type=str,
        default="",
        help="Comma-separated early-margin thresholds for the x-axis.",
    )
    parser.add_argument(
        "--kl_thresholds",
        type=str,
        default="",
        help="Comma-separated KL thresholds, one colored curve set per threshold.",
    )
    parser.add_argument(
        "--min_hits",
        type=int,
        default=1,
        help="Minimum number of matching rows required to trigger a block.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="phase3_margin_kl_precision_recall_profile",
    )
    return parser.parse_args()


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def load_profile(path: str, source_kind: str) -> Dict[str, pd.DataFrame]:
    full_df = pd.read_csv(path)
    block_df = (
        full_df.groupby(["step", "cycle_idx"], as_index=False)["block_rejected"]
        .max()
        .sort_values(["step", "cycle_idx"])
        .reset_index(drop=True)
    )

    usable_df = full_df[
        (full_df["is_bonus_position"] == 0)
        & (full_df["pre_final_features_available"] == 1)
    ].copy()
    if source_kind != "all":
        usable_df = usable_df[usable_df["source_kind"] == source_kind].copy()

    return {"blocks": block_df, "rows": usable_df}


def evaluate_threshold_grid(
    block_df: pd.DataFrame,
    row_df: pd.DataFrame,
    margin_thresholds: List[float],
    kl_thresholds: List[float],
    min_hits: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    total_blocks = len(block_df)
    rejected_blocks = int(block_df["block_rejected"].sum())
    accepted_blocks = total_blocks - rejected_blocks

    for kl_threshold in kl_thresholds:
        for margin_threshold in margin_thresholds:
            matching = row_df[
                (row_df["early_margin"] < margin_threshold)
                & (row_df["kl_early_draft"] >= kl_threshold)
            ]
            trigger_counts = (
                matching.groupby(["step", "cycle_idx"])
                .size()
                .rename("trigger_count")
                .reset_index()
            )
            merged = block_df.merge(trigger_counts, on=["step", "cycle_idx"], how="left")
            merged["trigger_count"] = merged["trigger_count"].fillna(0).astype(int)
            merged["triggered"] = merged["trigger_count"] >= min_hits

            tp = int(((merged["triggered"]) & (merged["block_rejected"] == 1)).sum())
            fp = int(((merged["triggered"]) & (merged["block_rejected"] == 0)).sum())
            tn = int(((~merged["triggered"]) & (merged["block_rejected"] == 0)).sum())
            fn = int(((~merged["triggered"]) & (merged["block_rejected"] == 1)).sum())

            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(2 * precision * recall, precision + recall)
            trigger_rate = safe_div(tp + fp, total_blocks)

            rows.append(
                {
                    "margin_threshold": margin_threshold,
                    "kl_threshold": kl_threshold,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "trigger_rate": trigger_rate,
                    "triggered_blocks": int(merged["triggered"].sum()),
                    "total_blocks": total_blocks,
                    "rejected_blocks": rejected_blocks,
                    "accepted_blocks": accepted_blocks,
                    "min_hits": min_hits,
                }
            )

    summary = pd.DataFrame(rows)
    return summary.sort_values(["kl_threshold", "margin_threshold"]).reset_index(drop=True)


def plot_profile(
    summary: pd.DataFrame,
    margin_thresholds: List[float],
    kl_thresholds: List[float],
    title: str,
    output_png: str,
):
    fig, ax_precision = plt.subplots(figsize=(11, 6))
    ax_recall = ax_precision.twinx()

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(kl_thresholds)))

    for color, kl_threshold in zip(colors, kl_thresholds):
        subset = summary[summary["kl_threshold"] == kl_threshold].sort_values("margin_threshold")
        ax_precision.plot(
            subset["margin_threshold"],
            subset["precision"],
            color=color,
            marker="o",
            linewidth=2,
        )
        ax_recall.plot(
            subset["margin_threshold"],
            subset["recall"],
            color=color,
            marker="s",
            linewidth=2,
            linestyle="--",
        )

    ax_precision.set_xlabel("early_margin threshold")
    ax_precision.set_ylabel("Precision")
    ax_recall.set_ylabel("Recall")
    ax_precision.set_ylim(0.0, 1.05)
    ax_recall.set_ylim(0.0, 1.05)
    ax_precision.set_xticks(margin_thresholds)
    ax_precision.set_xticklabels([f"{value:.3f}" for value in margin_thresholds])
    ax_precision.grid(True, linestyle="--", alpha=0.35)

    color_handles = [
        Line2D([0], [0], color=color, lw=2, marker="o", label=f"KL >= {kl_threshold:.3f}")
        for color, kl_threshold in zip(colors, kl_thresholds)
    ]
    metric_handles = [
        Line2D([0], [0], color="black", lw=2, marker="o", linestyle="-", label="Precision"),
        Line2D([0], [0], color="black", lw=2, marker="s", linestyle="--", label="Recall"),
    ]

    legend_colors = ax_precision.legend(handles=color_handles, title="KL threshold", loc="upper left")
    ax_precision.add_artist(legend_colors)
    ax_recall.legend(handles=metric_handles, title="Metric", loc="upper right")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    margin_thresholds = parse_float_list(args.margin_thresholds, DEFAULT_MARGIN_THRESHOLDS)
    kl_thresholds = parse_float_list(args.kl_thresholds, DEFAULT_KL_THRESHOLDS)

    input_dir = os.path.dirname(args.input_csv)
    output_png = args.output_png or os.path.join(input_dir, f"{args.output_prefix}.png")
    output_csv = args.output_csv or os.path.join(input_dir, f"{args.output_prefix}.csv")

    ensure_parent(output_png)
    ensure_parent(output_csv)

    data = load_profile(args.input_csv, args.source_kind)
    summary = evaluate_threshold_grid(
        block_df=data["blocks"],
        row_df=data["rows"],
        margin_thresholds=margin_thresholds,
        kl_thresholds=kl_thresholds,
        min_hits=args.min_hits,
    )
    summary.insert(0, "source_kind", args.source_kind)

    summary.to_csv(output_csv, index=False)

    plot_title = args.title or os.path.basename(args.input_csv)
    plot_profile(
        summary=summary,
        margin_thresholds=margin_thresholds,
        kl_thresholds=kl_thresholds,
        title=plot_title,
        output_png=output_png,
    )

    print(f"Wrote summary CSV to {output_csv}")
    print(f"Wrote figure PNG to {output_png}")


if __name__ == "__main__":
    main()
