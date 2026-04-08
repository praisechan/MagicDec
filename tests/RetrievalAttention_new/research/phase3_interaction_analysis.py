import argparse
import os
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MARGIN_BINS = [0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 1.01]
KL_BINS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 1.01]
MARGIN_SLICE_BINS = [-np.inf, 0.03, 0.10, np.inf]
MARGIN_SLICE_LABELS = ["low(<0.03)", "mid(0.03-0.10)", "high(>=0.10)"]
KL_SLICE_BINS = [-np.inf, 0.01, 0.03, np.inf]
KL_SLICE_LABELS = ["low(<0.01)", "moderate(0.01-0.03)", "high(>=0.03)"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explain the interaction between early margin and KL(early_verify || draft) in Phase 3."
    )
    parser.add_argument("--steps20_csv", type=str, required=True)
    parser.add_argument("--steps50_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_prefix", type=str, default="phase3_interaction")
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_profile(path: str, dataset_name: str) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    df = df[(df["is_bonus_position"] == 0) & (df["pre_final_features_available"] == 1)].copy()
    df["dataset"] = dataset_name
    df["block_id"] = df["step"].astype(str) + ":" + df["cycle_idx"].astype(str)

    accepted_prefix = df[df["source_kind"] == "accepted_prefix"].copy()
    accepted_prefix["margin_slice"] = pd.cut(
        accepted_prefix["early_margin"],
        bins=MARGIN_SLICE_BINS,
        labels=MARGIN_SLICE_LABELS,
        right=False,
    )
    accepted_prefix["kl_slice"] = pd.cut(
        accepted_prefix["kl_early_draft"],
        bins=KL_SLICE_BINS,
        labels=KL_SLICE_LABELS,
        right=False,
    )
    accepted_prefix["margin_bin"] = pd.cut(
        accepted_prefix["early_margin"],
        bins=MARGIN_BINS,
        include_lowest=True,
        right=False,
    )
    accepted_prefix["kl_bin"] = pd.cut(
        accepted_prefix["kl_early_draft"],
        bins=KL_BINS,
        include_lowest=True,
        right=False,
    )
    return {"all": df, "accepted_prefix": accepted_prefix}


def label_interval(interval) -> str:
    if pd.isna(interval):
        return "NA"
    return f"[{interval.left:.3f}, {interval.right:.3f})"


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def compute_conditional_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    base_rate = df["is_rejected_position"].mean()
    rows: List[Dict[str, object]] = []

    by_margin_kl = (
        df.groupby(["margin_slice", "kl_slice"], observed=False)["is_rejected_position"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    for row in by_margin_kl.itertuples(index=False):
        rows.append(
            {
                "dataset": dataset_name,
                "view": "token_margin_x_kl_slice",
                "group_a": row.margin_slice,
                "group_b": row.kl_slice,
                "token_count": int(row.count),
                "rejected_token_count": int(row.sum),
                "reject_rate": float(row.mean),
                "lift_vs_dataset_base": safe_div(float(row.mean), base_rate),
            }
        )

    by_margin_bin = (
        df.groupby(["margin_slice", "kl_bin"], observed=False)["is_rejected_position"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    for row in by_margin_bin.itertuples(index=False):
        rows.append(
            {
                "dataset": dataset_name,
                "view": "token_kl_within_margin_slice",
                "group_a": row.margin_slice,
                "group_b": label_interval(row.kl_bin),
                "token_count": int(row.count),
                "rejected_token_count": int(row.sum),
                "reject_rate": float(row.mean),
                "lift_vs_dataset_base": safe_div(float(row.mean), base_rate),
            }
        )

    by_kl_bin = (
        df.groupby(["kl_slice", "margin_bin"], observed=False)["is_rejected_position"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    for row in by_kl_bin.itertuples(index=False):
        rows.append(
            {
                "dataset": dataset_name,
                "view": "token_margin_within_kl_slice",
                "group_a": row.kl_slice,
                "group_b": label_interval(row.margin_bin),
                "token_count": int(row.count),
                "rejected_token_count": int(row.sum),
                "reject_rate": float(row.mean),
                "lift_vs_dataset_base": safe_div(float(row.mean), base_rate),
            }
        )

    heatmap = (
        df.groupby(["margin_bin", "kl_bin"], observed=False)["is_rejected_position"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    for row in heatmap.itertuples(index=False):
        rows.append(
            {
                "dataset": dataset_name,
                "view": "token_heatmap_bin",
                "group_a": label_interval(row.margin_bin),
                "group_b": label_interval(row.kl_bin),
                "token_count": int(row.count),
                "rejected_token_count": int(row.sum),
                "reject_rate": float(row.mean),
                "lift_vs_dataset_base": safe_div(float(row.mean), base_rate),
            }
        )
    return pd.DataFrame(rows)


def evaluate_block_trigger(block_df: pd.DataFrame, trigger_col: str) -> Dict[str, float]:
    tp = int(((block_df[trigger_col]) & (block_df["block_rejected"] == 1)).sum())
    fp = int(((block_df[trigger_col]) & (block_df["block_rejected"] == 0)).sum())
    tn = int(((~block_df[trigger_col]) & (block_df["block_rejected"] == 0)).sum())
    fn = int(((~block_df[trigger_col]) & (block_df["block_rejected"] == 1)).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "trigger_rate": safe_div(tp + fp, len(block_df)),
    }


def compute_block_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    cat_masks = {
        "low_margin": df["early_margin"] < 0.03,
        "moderate_kl": (df["kl_early_draft"] >= 0.01) & (df["kl_early_draft"] < 0.03),
        "high_kl": df["kl_early_draft"] >= 0.03,
        "low_margin_and_moderate_kl": (df["early_margin"] < 0.03)
        & (df["kl_early_draft"] >= 0.01)
        & (df["kl_early_draft"] < 0.03),
        "low_margin_and_high_kl": (df["early_margin"] < 0.03) & (df["kl_early_draft"] >= 0.03),
        "moderate_kl_not_low_margin": (df["early_margin"] >= 0.03)
        & (df["kl_early_draft"] >= 0.01)
        & (df["kl_early_draft"] < 0.03),
        "high_kl_not_low_margin": (df["early_margin"] >= 0.03) & (df["kl_early_draft"] >= 0.03),
    }

    block_df = df.groupby("block_id").agg(block_rejected=("block_rejected", "max"))
    for name, mask in cat_masks.items():
        block_df[f"{name}_count"] = df[mask].groupby("block_id").size().reindex(block_df.index, fill_value=0)
        block_df[f"{name}_any"] = block_df[f"{name}_count"] >= 1

    rows: List[Dict[str, object]] = []
    total_blocks = len(block_df)
    total_rejected = int(block_df["block_rejected"].sum())
    base_reject_rate = safe_div(total_rejected, total_blocks)

    for name in cat_masks:
        for threshold in (1, 2, 3):
            trigger_col = f"{name}_count_ge_{threshold}"
            block_df[trigger_col] = block_df[f"{name}_count"] >= threshold
            metrics = evaluate_block_trigger(block_df, trigger_col)
            rows.append(
                {
                    "dataset": dataset_name,
                    "view": "block_trigger",
                    "condition": name,
                    "threshold_kind": "count_ge",
                    "threshold_value": threshold,
                    "subset": "all_blocks",
                    "subset_size": total_blocks,
                    "subset_reject_rate": base_reject_rate,
                    "triggered_blocks": int(block_df[trigger_col].sum()),
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "tn": metrics["tn"],
                    "fn": metrics["fn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "trigger_rate": metrics["trigger_rate"],
                    "lift_vs_dataset_block_base": safe_div(metrics["precision"], base_reject_rate),
                }
            )

    conditional_subsets = {
        "within_low_margin_blocks": block_df["low_margin_any"],
        "within_high_kl_blocks": block_df["high_kl_any"],
    }
    subset_triggers = {
        "within_low_margin_blocks": ["low_margin_any", "low_margin_and_moderate_kl_any", "low_margin_and_high_kl_any"],
        "within_high_kl_blocks": ["high_kl_any", "low_margin_and_high_kl_any", "high_kl_not_low_margin_any"],
    }

    for subset_name, subset_mask in conditional_subsets.items():
        subset_df = block_df[subset_mask].copy()
        subset_reject_rate = safe_div(int(subset_df["block_rejected"].sum()), len(subset_df))
        for trigger_name in subset_triggers[subset_name]:
            metrics = evaluate_block_trigger(subset_df, trigger_name)
            rows.append(
                {
                    "dataset": dataset_name,
                    "view": "block_conditional_subset",
                    "condition": trigger_name,
                    "threshold_kind": "any",
                    "threshold_value": 1,
                    "subset": subset_name,
                    "subset_size": len(subset_df),
                    "subset_reject_rate": subset_reject_rate,
                    "triggered_blocks": int(subset_df[trigger_name].sum()),
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "tn": metrics["tn"],
                    "fn": metrics["fn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "trigger_rate": metrics["trigger_rate"],
                    "lift_vs_dataset_block_base": safe_div(metrics["precision"], base_reject_rate),
                }
            )

    for name in [
        "low_margin",
        "moderate_kl",
        "high_kl",
        "low_margin_and_moderate_kl",
        "low_margin_and_high_kl",
        "high_kl_not_low_margin",
    ]:
        dist = (
            block_df.groupby([f"{name}_count", "block_rejected"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for row in dist.itertuples(index=False):
            accepted_count = int(getattr(row, "_1", 0) if hasattr(row, "_1") else row[1])
            rejected_count = int(getattr(row, "_2", 0) if hasattr(row, "_2") else row[2])
            count_value = int(row[0])
            total = accepted_count + rejected_count
            rows.append(
                {
                    "dataset": dataset_name,
                    "view": "block_count_distribution",
                    "condition": name,
                    "threshold_kind": "exact_count",
                    "threshold_value": count_value,
                    "subset": "all_blocks",
                    "subset_size": total,
                    "subset_reject_rate": safe_div(rejected_count, total),
                    "triggered_blocks": total,
                    "tp": rejected_count,
                    "fp": accepted_count,
                    "tn": 0,
                    "fn": 0,
                    "precision": safe_div(rejected_count, total),
                    "recall": 0.0,
                    "trigger_rate": safe_div(total, total_blocks),
                    "lift_vs_dataset_block_base": safe_div(safe_div(rejected_count, total), base_reject_rate),
                }
            )

    return pd.DataFrame(rows)


def compute_mechanism_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    base_token_rate = df["is_rejected_position"].mean()
    block_df = df.groupby("block_id").agg(block_rejected=("block_rejected", "max"))
    base_block_rate = safe_div(int(block_df["block_rejected"].sum()), len(block_df))

    token_conditions = {
        "low_margin_only": (df["early_margin"] < 0.03) & (df["kl_early_draft"] < 0.01),
        "low_margin_plus_moderate_kl": (df["early_margin"] < 0.03)
        & (df["kl_early_draft"] >= 0.01)
        & (df["kl_early_draft"] < 0.03),
        "low_margin_plus_high_kl": (df["early_margin"] < 0.03) & (df["kl_early_draft"] >= 0.03),
        "high_kl_without_low_margin": (df["early_margin"] >= 0.03) & (df["kl_early_draft"] >= 0.03),
        "moderate_kl_without_low_margin": (df["early_margin"] >= 0.03)
        & (df["kl_early_draft"] >= 0.01)
        & (df["kl_early_draft"] < 0.03),
    }

    rows = []
    for name, mask in token_conditions.items():
        sub = df[mask]
        reject_rate = sub["is_rejected_position"].mean() if len(sub) else 0.0
        rows.append(
            {
                "dataset": dataset_name,
                "level": "token",
                "condition": name,
                "unit_count": len(sub),
                "positive_count": int(sub["is_rejected_position"].sum()),
                "rate": reject_rate,
                "lift_vs_base": safe_div(reject_rate, base_token_rate),
            }
        )

    block_conditions = {
        "any_low_margin": df["early_margin"] < 0.03,
        "any_moderate_kl": (df["kl_early_draft"] >= 0.01) & (df["kl_early_draft"] < 0.03),
        "any_low_margin_plus_moderate_kl": (df["early_margin"] < 0.03)
        & (df["kl_early_draft"] >= 0.01)
        & (df["kl_early_draft"] < 0.03),
        "any_low_margin_plus_high_kl": (df["early_margin"] < 0.03) & (df["kl_early_draft"] >= 0.03),
        "any_high_kl_without_low_margin": (df["early_margin"] >= 0.03) & (df["kl_early_draft"] >= 0.03),
        "two_low_margin_plus_high_kl": (df["early_margin"] < 0.03) & (df["kl_early_draft"] >= 0.03),
    }

    for name, mask in block_conditions.items():
        counts = df[mask].groupby("block_id").size().reindex(block_df.index, fill_value=0)
        threshold = 2 if name == "two_low_margin_plus_high_kl" else 1
        trigger = counts >= threshold
        tp = int((trigger & (block_df["block_rejected"] == 1)).sum())
        fp = int((trigger & (block_df["block_rejected"] == 0)).sum())
        reject_rate = safe_div(tp, int(trigger.sum()))
        rows.append(
            {
                "dataset": dataset_name,
                "level": "block",
                "condition": name,
                "unit_count": int(trigger.sum()),
                "positive_count": tp,
                "rate": reject_rate,
                "lift_vs_base": safe_div(reject_rate, base_block_rate),
            }
        )

    return pd.DataFrame(rows)


def midpoint_labels(edges: Iterable[float]) -> List[str]:
    edge_list = list(edges)
    return [f"{edge_list[i]:.3f}\n-{edge_list[i + 1]:.3f}" for i in range(len(edge_list) - 1)]


def heatmap_matrices(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    grouped = (
        df.groupby(["margin_bin", "kl_bin"], observed=False)["is_rejected_position"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    count_matrix = np.zeros((len(MARGIN_BINS) - 1, len(KL_BINS) - 1), dtype=float)
    rate_matrix = np.zeros((len(MARGIN_BINS) - 1, len(KL_BINS) - 1), dtype=float)
    for row in grouped.itertuples(index=False):
        margin_idx = list(df["margin_bin"].cat.categories).index(row.margin_bin)
        kl_idx = list(df["kl_bin"].cat.categories).index(row.kl_bin)
        count_matrix[margin_idx, kl_idx] = int(row.count)
        rate_matrix[margin_idx, kl_idx] = float(row.mean)
    return count_matrix, rate_matrix


def plot_heatmaps(datasets: Dict[str, pd.DataFrame], output_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), constrained_layout=True)
    xlabels = midpoint_labels(KL_BINS)
    ylabels = midpoint_labels(MARGIN_BINS)

    for row_idx, (dataset_name, df) in enumerate(datasets.items()):
        count_matrix, rate_matrix = heatmap_matrices(df)
        share_matrix = safe_div(count_matrix, count_matrix.sum())

        ax_rate = axes[row_idx, 0]
        ax_share = axes[row_idx, 1]

        im_rate = ax_rate.imshow(rate_matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=max(0.05, rate_matrix.max()))
        im_share = ax_share.imshow(share_matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(0.08, share_matrix.max()))

        for ax, title in (
            (ax_rate, f"{dataset_name}: reject rate by (margin, KL)"),
            (ax_share, f"{dataset_name}: token share by (margin, KL)"),
        ):
            ax.set_xticks(range(len(xlabels)))
            ax.set_xticklabels(xlabels)
            ax.set_yticks(range(len(ylabels)))
            ax.set_yticklabels(ylabels)
            ax.set_xlabel("KL(early_verify || draft) bin")
            ax.set_ylabel("early_margin bin")
            ax.set_title(title)

        for i in range(rate_matrix.shape[0]):
            for j in range(rate_matrix.shape[1]):
                ax_rate.text(j, i, f"{100.0 * rate_matrix[i, j]:.1f}%", ha="center", va="center", color="white", fontsize=8)
                ax_share.text(j, i, f"{count_matrix[i, j]:.0f}", ha="center", va="center", color="black", fontsize=8)

        fig.colorbar(im_rate, ax=ax_rate, fraction=0.046, pad=0.04)
        fig.colorbar(im_share, ax=ax_share, fraction=0.046, pad=0.04)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def line_series(df: pd.DataFrame, group_col: str, bin_col: str) -> Dict[str, pd.DataFrame]:
    result = {}
    grouped = (
        df.groupby([group_col, bin_col], observed=False)["is_rejected_position"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    for group_name, sub in grouped.groupby(group_col, observed=False):
        result[str(group_name)] = sub.copy()
    return result


def plot_conditional_lines(datasets: Dict[str, pd.DataFrame], output_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    markers = ["o", "s", "^"]

    for row_idx, (dataset_name, df) in enumerate(datasets.items()):
        kl_series = line_series(df, "margin_slice", "kl_bin")
        margin_series = line_series(df, "kl_slice", "margin_bin")
        ax_kl = axes[row_idx, 0]
        ax_margin = axes[row_idx, 1]

        for idx, label in enumerate(MARGIN_SLICE_LABELS):
            sub = kl_series[label]
            xs = [label_interval(x) for x in sub["kl_bin"]]
            ax_kl.plot(xs, sub["mean"] * 100.0, marker=markers[idx], linewidth=1.8, label=label)
        ax_kl.set_title(f"{dataset_name}: reject rate vs KL within margin slices")
        ax_kl.set_ylabel("Reject rate (%)")
        ax_kl.set_xlabel("KL(early_verify || draft) bin")
        ax_kl.tick_params(axis="x", rotation=35)
        ax_kl.grid(True, linestyle="--", alpha=0.4)
        ax_kl.legend(frameon=False)

        for idx, label in enumerate(KL_SLICE_LABELS):
            sub = margin_series[label]
            xs = [label_interval(x) for x in sub["margin_bin"]]
            ax_margin.plot(xs, sub["mean"] * 100.0, marker=markers[idx], linewidth=1.8, label=label)
        ax_margin.set_title(f"{dataset_name}: reject rate vs margin within KL slices")
        ax_margin.set_ylabel("Reject rate (%)")
        ax_margin.set_xlabel("early_margin bin")
        ax_margin.tick_params(axis="x", rotation=35)
        ax_margin.grid(True, linestyle="--", alpha=0.4)
        ax_margin.legend(frameon=False)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.dirname(args.steps20_csv)
    ensure_dir(output_dir)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    datasets = {
        "steps20": load_profile(args.steps20_csv, "steps20"),
        "steps50": load_profile(args.steps50_csv, "steps50"),
    }
    accepted_prefix_datasets = {
        dataset_name: bundle["accepted_prefix"] for dataset_name, bundle in datasets.items()
    }

    conditional_summary = pd.concat(
        [compute_conditional_summary(df, dataset_name) for dataset_name, df in accepted_prefix_datasets.items()],
        ignore_index=True,
    )
    block_summary = pd.concat(
        [compute_block_summary(df, dataset_name) for dataset_name, df in accepted_prefix_datasets.items()],
        ignore_index=True,
    )
    mechanism_summary = pd.concat(
        [compute_mechanism_summary(df, dataset_name) for dataset_name, df in accepted_prefix_datasets.items()],
        ignore_index=True,
    )

    conditional_csv = os.path.join(output_dir, f"{args.output_prefix}_accepted_prefix_conditional_summary.csv")
    block_csv = os.path.join(output_dir, f"{args.output_prefix}_accepted_prefix_block_summary.csv")
    mechanism_csv = os.path.join(output_dir, f"{args.output_prefix}_accepted_prefix_mechanism_summary.csv")
    heatmap_png = os.path.join(output_dir, f"{args.output_prefix}_accepted_prefix_heatmaps.png")
    lines_png = os.path.join(output_dir, f"{args.output_prefix}_accepted_prefix_conditional_lines.png")

    write_csv(conditional_summary, conditional_csv)
    write_csv(block_summary, block_csv)
    write_csv(mechanism_summary, mechanism_csv)
    plot_heatmaps(accepted_prefix_datasets, heatmap_png)
    plot_conditional_lines(accepted_prefix_datasets, lines_png)

    print(f"conditional_csv: {conditional_csv}")
    print(f"block_csv: {block_csv}")
    print(f"mechanism_csv: {mechanism_csv}")
    print(f"heatmap_png: {heatmap_png}")
    print(f"lines_png: {lines_png}")


if __name__ == "__main__":
    main()
