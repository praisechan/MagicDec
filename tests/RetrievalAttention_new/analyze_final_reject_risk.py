import argparse
import csv
import json
import math
import os
import sys


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from MagicDec.tests.RetrievalAttention_new.riskgate_utils import (
    RECOMMENDED_ASSISTED_RULE,
    RECOMMENDED_DRAFT_RULE,
    compute_assisted_risk_score,
    compute_risk_score,
)


DRAFT_RULES = [
    "min_gap",
    "mean_margin",
    "last_margin",
    "min_margin_mean_margin",
    "min_margin_low_count",
    "min_margin_early_position",
    "min_margin_mean_early",
]

ASSISTED_RULES = [
    "normal_accept_ratio",
    "normal_accept_plus_min_margin",
]

TOKEN_RULES = [
    "token_margin",
    "token_top2_prob",
    "token_entropy",
    "token_margin_entropy",
    "token_margin_early_position",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze richer final-reject profiling logs.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--T_low", type=float, default=0.05)
    parser.add_argument("--T_high", type=float, default=0.20)
    return parser.parse_args()


def maybe_number(value):
    if value is None or value == "":
        return value
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except (TypeError, ValueError):
        return value


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [{key: maybe_number(value) for key, value in row.items()} for row in csv.DictReader(f)]


def compute_binary_metrics(labels, selected):
    total = len(labels)
    positives = sum(labels)
    selected_count = sum(selected)
    tp = sum(1 for label, flag in zip(labels, selected) if label == 1 and flag == 1)
    fp = sum(1 for label, flag in zip(labels, selected) if label == 0 and flag == 1)
    fn = positives - tp
    precision = float(tp / selected_count) if selected_count > 0 else 0.0
    recall = float(tp / positives) if positives > 0 else 0.0
    trigger_rate = float(selected_count / total) if total > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "selected_count": selected_count,
        "precision": precision,
        "recall": recall,
        "trigger_rate": trigger_rate,
    }


def choose_by_trigger(rows, target_trigger):
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            abs(row["trigger_rate"] - target_trigger),
            -row["precision"],
            -row["recall"],
            row["selected_count"],
        ),
    )[0]


def choose_by_recall(rows, target_recall):
    eligible = [row for row in rows if row["recall"] >= target_recall]
    if eligible:
        return sorted(
            eligible,
            key=lambda row: (
                -row["precision"],
                abs(row["recall"] - target_recall),
                row["trigger_rate"],
            ),
        )[0]
    return sorted(
        rows,
        key=lambda row: (
            abs(row["recall"] - target_recall),
            -row["precision"],
            row["trigger_rate"],
        ),
    )[0]


def sweep_thresholds(scores, labels, eligible_mask, rule_name):
    unique_thresholds = sorted(set(scores), reverse=True)
    results = []
    for threshold in unique_thresholds:
        selected = [
            1 if (eligible_mask[idx] and scores[idx] >= threshold) else 0
            for idx in range(len(scores))
        ]
        metrics = compute_binary_metrics(labels, selected)
        results.append(
            {
                "rule_name": rule_name,
                "threshold": float(threshold),
                **metrics,
            }
        )

    empty_metrics = compute_binary_metrics(labels, [0] * len(scores))
    results.append(
        {
            "rule_name": rule_name,
            "threshold": float("inf"),
            **empty_metrics,
        }
    )
    return results


def token_rule_score(row, rule_name):
    margin_risk = 1.0 - float(row["margin"])
    if rule_name == "token_margin":
        return margin_risk
    if rule_name == "token_top2_prob":
        return float(row["top2_prob"])
    if rule_name == "token_entropy":
        return float(row["entropy"])
    if rule_name == "token_margin_entropy":
        return margin_risk + 0.2 * float(row["entropy"])
    if rule_name == "token_margin_early_position":
        return margin_risk + 0.3 * (1.0 - float(row["token_position_norm"]))
    raise KeyError(f"Unknown token rule: {rule_name}")


def summarize_rows(rows, keys):
    return [{key: row[key] for key in keys} for row in rows]


def main():
    args = parse_args()
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    block_csv = os.path.join(run_dir, "block_features.csv")
    token_csv = os.path.join(run_dir, "token_features.csv")
    summary_json = os.path.join(run_dir, "run_summary.json")

    if not os.path.exists(block_csv):
        raise FileNotFoundError(f"Missing block_features.csv in {run_dir}")
    if not os.path.exists(token_csv):
        raise FileNotFoundError(f"Missing token_features.csv in {run_dir}")

    block_rows = load_csv_rows(block_csv)
    token_rows = load_csv_rows(token_csv)
    run_summary = {}
    if os.path.exists(summary_json):
        with open(summary_json, "r", encoding="utf-8") as f:
            run_summary = json.load(f)

    valid_blocks = [row for row in block_rows if int(row["block_label_valid"]) == 1]
    if not valid_blocks:
        raise RuntimeError("No valid block rows found for analysis.")

    block_labels = [int(row["block_final_reject"]) for row in valid_blocks]
    block_eligible = [float(row["block_min_margin"]) <= args.T_high for row in valid_blocks]

    baseline_selected = [
        1 if eligible and float(row["block_min_margin"]) < args.T_low else 0
        for row, eligible in zip(valid_blocks, block_eligible)
    ]
    baseline_metrics = compute_binary_metrics(block_labels, baseline_selected)
    baseline_metrics["rule_name"] = "baseline_min_gap"
    baseline_metrics["threshold"] = args.T_low

    block_candidate_rows = []
    best_draft_by_trigger = None
    best_draft_by_recall = None
    best_assisted_by_trigger = None
    best_assisted_by_recall = None

    for rule_name in DRAFT_RULES:
        scores = [compute_risk_score(row, rule_name) for row in valid_blocks]
        threshold_rows = sweep_thresholds(scores, block_labels, block_eligible, rule_name)
        matched_trigger = choose_by_trigger(threshold_rows, baseline_metrics["trigger_rate"])
        matched_recall = choose_by_recall(threshold_rows, baseline_metrics["recall"])
        block_candidate_rows.append(
            {
                "rule_family": "draft_only",
                "comparison": "matched_trigger",
                **matched_trigger,
            }
        )
        block_candidate_rows.append(
            {
                "rule_family": "draft_only",
                "comparison": "matched_recall",
                **matched_recall,
            }
        )
        if best_draft_by_trigger is None or (
            matched_trigger["precision"],
            matched_trigger["recall"],
            -matched_trigger["trigger_rate"],
        ) > (
            best_draft_by_trigger["precision"],
            best_draft_by_trigger["recall"],
            -best_draft_by_trigger["trigger_rate"],
        ):
            best_draft_by_trigger = {"rule_family": "draft_only", "comparison": "matched_trigger", **matched_trigger}
        if best_draft_by_recall is None or (
            matched_recall["precision"],
            -matched_recall["trigger_rate"],
            matched_recall["recall"],
        ) > (
            best_draft_by_recall["precision"],
            -best_draft_by_recall["trigger_rate"],
            best_draft_by_recall["recall"],
        ):
            best_draft_by_recall = {"rule_family": "draft_only", "comparison": "matched_recall", **matched_recall}

    for rule_name in ASSISTED_RULES:
        scores = [compute_assisted_risk_score(row, rule_name) for row in valid_blocks]
        threshold_rows = sweep_thresholds(scores, block_labels, block_eligible, rule_name)
        matched_trigger = choose_by_trigger(threshold_rows, baseline_metrics["trigger_rate"])
        matched_recall = choose_by_recall(threshold_rows, baseline_metrics["recall"])
        block_candidate_rows.append(
            {
                "rule_family": "assisted",
                "comparison": "matched_trigger",
                **matched_trigger,
            }
        )
        block_candidate_rows.append(
            {
                "rule_family": "assisted",
                "comparison": "matched_recall",
                **matched_recall,
            }
        )
        if best_assisted_by_trigger is None or (
            matched_trigger["precision"],
            matched_trigger["recall"],
            -matched_trigger["trigger_rate"],
        ) > (
            best_assisted_by_trigger["precision"],
            best_assisted_by_trigger["recall"],
            -best_assisted_by_trigger["trigger_rate"],
        ):
            best_assisted_by_trigger = {"rule_family": "assisted", "comparison": "matched_trigger", **matched_trigger}
        if best_assisted_by_recall is None or (
            matched_recall["precision"],
            -matched_recall["trigger_rate"],
            matched_recall["recall"],
        ) > (
            best_assisted_by_recall["precision"],
            -best_assisted_by_recall["trigger_rate"],
            best_assisted_by_recall["recall"],
        ):
            best_assisted_by_recall = {"rule_family": "assisted", "comparison": "matched_recall", **matched_recall}

    valid_tokens = [row for row in token_rows if int(row["token_label_valid"]) == 1]
    token_labels = [int(row["token_final_reject"]) for row in valid_tokens]
    baseline_token_selected = [1 if float(row["margin"]) < args.T_low else 0 for row in valid_tokens]
    baseline_token_metrics = compute_binary_metrics(token_labels, baseline_token_selected)
    baseline_token_metrics["rule_name"] = "token_margin"
    baseline_token_metrics["threshold"] = args.T_low

    token_candidate_rows = []
    best_token_by_trigger = None
    for rule_name in TOKEN_RULES:
        scores = [token_rule_score(row, rule_name) for row in valid_tokens]
        threshold_rows = sweep_thresholds(scores, token_labels, [True] * len(valid_tokens), rule_name)
        matched_trigger = choose_by_trigger(threshold_rows, baseline_token_metrics["trigger_rate"])
        token_candidate_rows.append({"comparison": "matched_trigger", **matched_trigger})
        if best_token_by_trigger is None or (
            matched_trigger["precision"],
            matched_trigger["recall"],
            -matched_trigger["trigger_rate"],
        ) > (
            best_token_by_trigger["precision"],
            best_token_by_trigger["recall"],
            -best_token_by_trigger["trigger_rate"],
        ):
            best_token_by_trigger = {"comparison": "matched_trigger", **matched_trigger}

    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    block_metrics_csv = os.path.join(analysis_dir, "block_rule_metrics.csv")
    token_metrics_csv = os.path.join(analysis_dir, "token_rule_metrics.csv")
    recommendation_json = os.path.join(analysis_dir, "recommended_rule.json")
    summary_out_json = os.path.join(analysis_dir, "analysis_summary.json")

    with open(block_metrics_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(block_candidate_rows[0].keys()) if block_candidate_rows else []
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(block_candidate_rows)

    with open(token_metrics_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(token_candidate_rows[0].keys()) if token_candidate_rows else []
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(token_candidate_rows)

    recommendation = {
        "baseline_block": baseline_metrics,
        "baseline_token": baseline_token_metrics,
        "best_draft_matched_trigger": best_draft_by_trigger,
        "best_draft_matched_recall": best_draft_by_recall,
        "best_assisted_matched_trigger": best_assisted_by_trigger,
        "best_assisted_matched_recall": best_assisted_by_recall,
        "best_token_matched_trigger": best_token_by_trigger,
        "recommended_draft_rule_name": (
            best_draft_by_trigger["rule_name"] if best_draft_by_trigger is not None else RECOMMENDED_DRAFT_RULE
        ),
        "recommended_draft_threshold": (
            best_draft_by_trigger["threshold"] if best_draft_by_trigger is not None else math.inf
        ),
        "recommended_assisted_rule_name": (
            best_assisted_by_trigger["rule_name"]
            if best_assisted_by_trigger is not None
            else RECOMMENDED_ASSISTED_RULE
        ),
        "recommended_assisted_threshold": (
            best_assisted_by_trigger["threshold"] if best_assisted_by_trigger is not None else math.inf
        ),
    }

    with open(recommendation_json, "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)

    with open(summary_out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "run_summary": run_summary,
                "baseline_block": baseline_metrics,
                "baseline_token": baseline_token_metrics,
                "best_draft_matched_trigger": best_draft_by_trigger,
                "best_draft_matched_recall": best_draft_by_recall,
                "best_assisted_matched_trigger": best_assisted_by_trigger,
                "best_assisted_matched_recall": best_assisted_by_recall,
                "best_token_matched_trigger": best_token_by_trigger,
                "top_block_rows": summarize_rows(
                    sorted(
                        block_candidate_rows,
                        key=lambda row: (
                            row["comparison"] != "matched_trigger",
                            row["rule_family"] != "draft_only",
                            -row["precision"],
                            -row["recall"],
                        ),
                    )[:10],
                    list(block_candidate_rows[0].keys()) if block_candidate_rows else [],
                ),
                "top_token_rows": summarize_rows(
                    sorted(token_candidate_rows, key=lambda row: (-row["precision"], -row["recall"]))[:5],
                    list(token_candidate_rows[0].keys()) if token_candidate_rows else [],
                ),
            },
            f,
            indent=2,
        )

    print(f"run_dir: {run_dir}")
    print(
        "baseline_block:"
        f" precision={baseline_metrics['precision']:.4f}"
        f" recall={baseline_metrics['recall']:.4f}"
        f" trigger_rate={baseline_metrics['trigger_rate']:.4f}"
        f" threshold(min_margin<{args.T_low})"
    )
    if best_draft_by_trigger is not None:
        print(
            "best_draft_matched_trigger:"
            f" rule={best_draft_by_trigger['rule_name']}"
            f" threshold={best_draft_by_trigger['threshold']:.6f}"
            f" precision={best_draft_by_trigger['precision']:.4f}"
            f" recall={best_draft_by_trigger['recall']:.4f}"
            f" trigger_rate={best_draft_by_trigger['trigger_rate']:.4f}"
        )
    if best_assisted_by_trigger is not None:
        print(
            "best_assisted_matched_trigger:"
            f" rule={best_assisted_by_trigger['rule_name']}"
            f" threshold={best_assisted_by_trigger['threshold']:.6f}"
            f" precision={best_assisted_by_trigger['precision']:.4f}"
            f" recall={best_assisted_by_trigger['recall']:.4f}"
            f" trigger_rate={best_assisted_by_trigger['trigger_rate']:.4f}"
        )
    if best_token_by_trigger is not None:
        print(
            "best_token_matched_trigger:"
            f" rule={best_token_by_trigger['rule_name']}"
            f" threshold={best_token_by_trigger['threshold']:.6f}"
            f" precision={best_token_by_trigger['precision']:.4f}"
            f" recall={best_token_by_trigger['recall']:.4f}"
            f" trigger_rate={best_token_by_trigger['trigger_rate']:.4f}"
        )
    print(f"block_metrics_csv: {block_metrics_csv}")
    print(f"token_metrics_csv: {token_metrics_csv}")
    print(f"recommendation_json: {recommendation_json}")
    print(f"analysis_summary_json: {summary_out_json}")


if __name__ == "__main__":
    main()
