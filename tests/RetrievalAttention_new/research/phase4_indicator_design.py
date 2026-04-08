import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4 robustness-aware pre-final indicator design centered on accepted-prefix margin+KL."
    )
    parser.add_argument("--steps20_csv", type=str, required=True)
    parser.add_argument("--steps50_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_prefix", type=str, default="phase4_indicator")
    return parser.parse_args()


def read_rows(path: str) -> List[Dict[str, str]]:
    csv.field_size_limit(1 << 30)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def to_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(value)


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def odds_ratio(tp: int, fp: int, tn: int, fn: int) -> float:
    # Haldane-Anscombe correction avoids division-by-zero explosions for pure rules.
    return ((tp + 0.5) * (tn + 0.5)) / ((fp + 0.5) * (fn + 0.5))


def block_key(row: Dict[str, str]) -> Tuple[int, int]:
    return int(row["step"]), int(row["cycle_idx"])


def block_id_from_key(key: Tuple[int, int]) -> str:
    return f"{key[0]}:{key[1]}"


def usable_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if row["is_bonus_position"] == "0" and row["pre_final_features_available"] == "1"
    ]


def sorted_values(rows: Iterable[Dict[str, str]], key: str, predicate: Callable[[Dict[str, str]], bool]) -> List[float]:
    values = [to_float(row, key, 0.0) for row in rows if predicate(row)]
    values.sort(reverse=True)
    return values


def kth_value(values: List[float], rank: int) -> float:
    return values[rank - 1] if len(values) >= rank else 0.0


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def count_if(rows: Iterable[Dict[str, str]], predicate: Callable[[Dict[str, str]], bool]) -> int:
    return sum(1 for row in rows if predicate(row))


def build_dataset(dataset_name: str, path: str) -> Dict[str, object]:
    grouped: Dict[Tuple[int, int], List[Dict[str, str]]] = defaultdict(list)
    for row in read_rows(path):
        grouped[block_key(row)].append(row)

    block_features: List[Dict[str, object]] = []
    token_summary_rows: List[Dict[str, object]] = []

    accepted_token_total = 0
    accepted_token_reject_total = 0
    category_totals = {
        "low_margin_only": [0, 0],
        "low_margin_plus_moderate_kl": [0, 0],
        "low_margin_plus_high_kl": [0, 0],
        "moderate_kl_without_low_margin": [0, 0],
        "high_kl_without_low_margin": [0, 0],
        "safe_other": [0, 0],
    }

    for key, rows in sorted(grouped.items()):
        filtered = usable_rows(rows)
        accepted_prefix_rows = [row for row in filtered if row["source_kind"] == "accepted_prefix"]
        early_bonus_rows = [row for row in filtered if row["source_kind"] == "early_bonus"]
        rejected = int(rows[0]["block_rejected"])

        ap_row_count = len(accepted_prefix_rows)
        ap_low_margin_count_003 = count_if(accepted_prefix_rows, lambda row: to_float(row, "early_margin", 1.0) < 0.03)
        ap_low_margin_count_005 = count_if(accepted_prefix_rows, lambda row: to_float(row, "early_margin", 1.0) < 0.05)

        ap_mod_count_003 = count_if(
            accepted_prefix_rows,
            lambda row: to_float(row, "early_margin", 1.0) < 0.03 and 0.01 <= to_float(row, "kl_early_draft", 0.0) < 0.03,
        )
        ap_high_count_003 = count_if(
            accepted_prefix_rows,
            lambda row: to_float(row, "early_margin", 1.0) < 0.03 and to_float(row, "kl_early_draft", 0.0) >= 0.03,
        )
        ap_mod_count_005 = count_if(
            accepted_prefix_rows,
            lambda row: to_float(row, "early_margin", 1.0) < 0.05 and 0.01 <= to_float(row, "kl_early_draft", 0.0) < 0.03,
        )
        ap_high_count_005 = count_if(
            accepted_prefix_rows,
            lambda row: to_float(row, "early_margin", 1.0) < 0.05 and to_float(row, "kl_early_draft", 0.0) >= 0.03,
        )
        ap_drift_count_003 = ap_mod_count_003 + ap_high_count_003
        ap_drift_count_005 = ap_mod_count_005 + ap_high_count_005

        low_margin_kl_003 = sorted_values(
            accepted_prefix_rows,
            "kl_early_draft",
            lambda row: to_float(row, "early_margin", 1.0) < 0.03,
        )
        low_margin_kl_005 = sorted_values(
            accepted_prefix_rows,
            "kl_early_draft",
            lambda row: to_float(row, "early_margin", 1.0) < 0.05,
        )

        block_features.append(
            {
                "dataset": dataset_name,
                "block_id": block_id_from_key(key),
                "step": key[0],
                "cycle_idx": key[1],
                "block_rejected": rejected,
                "accepted_prefix_rows": ap_row_count,
                "early_bonus_rows": len(early_bonus_rows),
                "ap_low_margin_count_lt_0p03": ap_low_margin_count_003,
                "ap_low_margin_count_lt_0p05": ap_low_margin_count_005,
                "ap_low_margin_mod_kl_count_lt_0p03": ap_mod_count_003,
                "ap_low_margin_high_kl_count_lt_0p03": ap_high_count_003,
                "ap_low_margin_drift_count_lt_0p03": ap_drift_count_003,
                "ap_low_margin_mod_kl_frac_lt_0p03": safe_div(ap_mod_count_003, ap_row_count),
                "ap_low_margin_high_kl_frac_lt_0p03": safe_div(ap_high_count_003, ap_row_count),
                "ap_low_margin_drift_frac_lt_0p03": safe_div(ap_drift_count_003, ap_row_count),
                "ap_low_margin_mod_kl_count_lt_0p05": ap_mod_count_005,
                "ap_low_margin_high_kl_count_lt_0p05": ap_high_count_005,
                "ap_low_margin_drift_count_lt_0p05": ap_drift_count_005,
                "ap_low_margin_mod_kl_frac_lt_0p05": safe_div(ap_mod_count_005, ap_row_count),
                "ap_low_margin_high_kl_frac_lt_0p05": safe_div(ap_high_count_005, ap_row_count),
                "ap_low_margin_drift_frac_lt_0p05": safe_div(ap_drift_count_005, ap_row_count),
                "ap_low_margin_top1_kl_lt_0p03": kth_value(low_margin_kl_003, 1),
                "ap_low_margin_top2_kl_lt_0p03": kth_value(low_margin_kl_003, 2),
                "ap_low_margin_top3_kl_lt_0p03": kth_value(low_margin_kl_003, 3),
                "ap_low_margin_q90_kl_lt_0p03": quantile(low_margin_kl_003, 0.90),
                "ap_low_margin_top1_kl_lt_0p05": kth_value(low_margin_kl_005, 1),
                "ap_low_margin_top2_kl_lt_0p05": kth_value(low_margin_kl_005, 2),
                "ap_low_margin_top3_kl_lt_0p05": kth_value(low_margin_kl_005, 3),
                "ap_low_margin_q90_kl_lt_0p05": quantile(low_margin_kl_005, 0.90),
                "early_bonus_fragile_count_pos_ge_0": count_if(
                    early_bonus_rows,
                    lambda row: to_float(row, "early_margin", 1.0) < 0.05 and to_float(row, "early_entropy", 0.0) > 3.5,
                ),
                "early_bonus_fragile_count_pos_ge_4": count_if(
                    early_bonus_rows,
                    lambda row: to_int(row, "position", 0) >= 4
                    and to_float(row, "early_margin", 1.0) < 0.05
                    and to_float(row, "early_entropy", 0.0) > 3.5,
                ),
                "early_bonus_fragile_count_pos_ge_8": count_if(
                    early_bonus_rows,
                    lambda row: to_int(row, "position", 0) >= 8
                    and to_float(row, "early_margin", 1.0) < 0.05
                    and to_float(row, "early_entropy", 0.0) > 3.5,
                ),
            }
        )

        for row in accepted_prefix_rows:
            accepted_token_total += 1
            token_rejected = int(row["is_rejected_position"])
            accepted_token_reject_total += token_rejected

            margin = to_float(row, "early_margin", 1.0)
            kl = to_float(row, "kl_early_draft", 0.0)
            if margin < 0.03 and kl < 0.01:
                category = "low_margin_only"
            elif margin < 0.03 and kl < 0.03:
                category = "low_margin_plus_moderate_kl"
            elif margin < 0.03:
                category = "low_margin_plus_high_kl"
            elif kl >= 0.03:
                category = "high_kl_without_low_margin"
            elif kl >= 0.01:
                category = "moderate_kl_without_low_margin"
            else:
                category = "safe_other"
            category_totals[category][0] += 1
            category_totals[category][1] += token_rejected

    base_token_rate = safe_div(accepted_token_reject_total, accepted_token_total)
    for category, (unit_count, rejected_count) in category_totals.items():
        rate = safe_div(rejected_count, unit_count)
        token_summary_rows.append(
            {
                "dataset": dataset_name,
                "category": category,
                "token_count": unit_count,
                "rejected_token_count": rejected_count,
                "reject_rate": rate,
                "lift_vs_dataset_base": safe_div(rate, base_token_rate),
            }
        )

    return {
        "dataset_name": dataset_name,
        "block_features": block_features,
        "token_summary": token_summary_rows,
        "base_block_reject_rate": safe_div(
            sum(int(row["block_rejected"]) for row in block_features),
            len(block_features),
        ),
    }


def compute_metrics(blocks: List[Dict[str, object]], trigger_fn: Callable[[Dict[str, object]], bool]) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for block in blocks:
        triggered = trigger_fn(block)
        rejected = int(block["block_rejected"])
        if triggered and rejected:
            tp += 1
        elif triggered and not rejected:
            fp += 1
        elif rejected:
            fn += 1
        else:
            tn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    trigger_rate = safe_div(tp + fp, len(blocks))
    base_rate = safe_div(tp + fn, len(blocks))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trigger_rate": trigger_rate,
        "lift_vs_block_base": safe_div(precision, base_rate),
        "odds_ratio": odds_ratio(tp, fp, tn, fn),
    }


def build_rule_specs() -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []

    for margin_label in ("0p03", "0p05"):
        drift_count_key = f"ap_low_margin_drift_count_lt_{margin_label}"
        high_count_key = f"ap_low_margin_high_kl_count_lt_{margin_label}"
        drift_frac_key = f"ap_low_margin_drift_frac_lt_{margin_label}"
        top2_key = f"ap_low_margin_top2_kl_lt_{margin_label}"
        q90_key = f"ap_low_margin_q90_kl_lt_{margin_label}"

        for threshold in (1, 2, 3):
            specs.append(
                {
                    "family": "accepted_drift_count",
                    "rule_name": f"accepted_drift_count|margin<{margin_label}|count>={threshold}",
                    "trigger_fn": lambda block, key=drift_count_key, threshold=threshold: block[key] >= threshold,
                    "deployable": 1,
                    "core_signal": "accepted_prefix_interaction",
                }
            )
            specs.append(
                {
                    "family": "accepted_high_count",
                    "rule_name": f"accepted_high_count|margin<{margin_label}|count>={threshold}",
                    "trigger_fn": lambda block, key=high_count_key, threshold=threshold: block[key] >= threshold,
                    "deployable": 1,
                    "core_signal": "accepted_prefix_interaction",
                }
            )

        for high_weight in (2, 3):
            for threshold in (2, 3, 4):
                specs.append(
                    {
                        "family": "accepted_weighted_score",
                        "rule_name": (
                            f"accepted_weighted_score|margin<{margin_label}|score={high_weight}*high+mod|>={threshold}"
                        ),
                        "trigger_fn": lambda block, margin_label=margin_label, high_weight=high_weight, threshold=threshold: (
                            high_weight * block[f"ap_low_margin_high_kl_count_lt_{margin_label}"]
                            + (
                                block[f"ap_low_margin_drift_count_lt_{margin_label}"]
                                - block[f"ap_low_margin_high_kl_count_lt_{margin_label}"]
                            )
                        )
                        >= threshold,
                        "deployable": 1,
                        "core_signal": "accepted_prefix_interaction",
                    }
                )

        for high_threshold in (1, 2):
            for drift_threshold in (2, 3):
                specs.append(
                    {
                        "family": "accepted_high_or_drift",
                        "rule_name": (
                            f"accepted_high_or_drift|margin<{margin_label}|high>={high_threshold}|drift>={drift_threshold}"
                        ),
                        "trigger_fn": lambda block, margin_label=margin_label, high_threshold=high_threshold, drift_threshold=drift_threshold: (
                            block[f"ap_low_margin_high_kl_count_lt_{margin_label}"] >= high_threshold
                            or block[f"ap_low_margin_drift_count_lt_{margin_label}"] >= drift_threshold
                        ),
                        "deployable": 1,
                        "core_signal": "accepted_prefix_interaction",
                    }
                )

        for frac_threshold in (0.04, 0.08, 0.12):
            for min_count in (1, 2):
                specs.append(
                    {
                        "family": "accepted_drift_fraction",
                        "rule_name": (
                            f"accepted_drift_fraction|margin<{margin_label}|frac>={frac_threshold:.2f}|count>={min_count}"
                        ),
                        "trigger_fn": lambda block, frac_key=drift_frac_key, count_key=drift_count_key, frac_threshold=frac_threshold, min_count=min_count: (
                            block[count_key] >= min_count and block[frac_key] >= frac_threshold
                        ),
                        "deployable": 1,
                        "core_signal": "accepted_prefix_interaction",
                    }
                )

        for top_threshold in (0.02, 0.03, 0.05):
            specs.append(
                {
                    "family": "accepted_top2_kl",
                    "rule_name": f"accepted_top2_kl|margin<{margin_label}|top2>={top_threshold:.2f}",
                    "trigger_fn": lambda block, top2_key=top2_key, top_threshold=top_threshold: block[top2_key] >= top_threshold,
                    "deployable": 1,
                    "core_signal": "accepted_prefix_interaction",
                }
            )
            specs.append(
                {
                    "family": "accepted_q90_kl",
                    "rule_name": f"accepted_q90_kl|margin<{margin_label}|q90>={top_threshold:.2f}",
                    "trigger_fn": lambda block, q90_key=q90_key, top_threshold=top_threshold: block[q90_key] >= top_threshold,
                    "deployable": 1,
                    "core_signal": "accepted_prefix_interaction",
                }
            )

    for margin_label in ("0p03", "0p05"):
        for interaction_family, interaction_template in (
            ("accepted_high_count", lambda block, margin_label=margin_label, n=2: block[f"ap_low_margin_high_kl_count_lt_{margin_label}"] >= n),
            ("accepted_drift_count", lambda block, margin_label=margin_label, n=2: block[f"ap_low_margin_drift_count_lt_{margin_label}"] >= n),
            (
                "accepted_weighted_score",
                lambda block, margin_label=margin_label: (
                    2 * block[f"ap_low_margin_high_kl_count_lt_{margin_label}"]
                    + (
                        block[f"ap_low_margin_drift_count_lt_{margin_label}"]
                        - block[f"ap_low_margin_high_kl_count_lt_{margin_label}"]
                    )
                ) >= 3,
            ),
        ):
            for bonus_key, bonus_label in (
                ("early_bonus_fragile_count_pos_ge_0", "bonus_ge0"),
                ("early_bonus_fragile_count_pos_ge_4", "bonus_ge4"),
                ("early_bonus_fragile_count_pos_ge_8", "bonus_ge8"),
            ):
                for bonus_count in (1, 2, 3):
                    specs.append(
                        {
                            "family": "source_or",
                            "rule_name": (
                                f"source_or|{interaction_family}|margin<{margin_label}|or|{bonus_label}_count>={bonus_count}"
                            ),
                            "trigger_fn": lambda block, interaction_template=interaction_template, bonus_key=bonus_key, bonus_count=bonus_count: (
                                interaction_template(block) or block[bonus_key] >= bonus_count
                            ),
                            "deployable": 1,
                            "core_signal": "accepted_prefix_interaction_plus_optional_bonus",
                        }
                    )

    return specs


def evaluate_rule_specs(datasets: Dict[str, Dict[str, object]], rule_specs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for spec in rule_specs:
        per_dataset: Dict[str, Dict[str, float]] = {}
        for dataset_name, dataset in datasets.items():
            per_dataset[dataset_name] = compute_metrics(dataset["block_features"], spec["trigger_fn"])

        row = {
            "family": spec["family"],
            "rule_name": spec["rule_name"],
            "deployable": spec["deployable"],
            "core_signal": spec["core_signal"],
        }
        for dataset_name, metrics in per_dataset.items():
            for key, value in metrics.items():
                row[f"{dataset_name}_{key}"] = value

        row["min_precision"] = min(per_dataset["steps20"]["precision"], per_dataset["steps50"]["precision"])
        row["min_recall"] = min(per_dataset["steps20"]["recall"], per_dataset["steps50"]["recall"])
        row["min_f1"] = min(per_dataset["steps20"]["f1"], per_dataset["steps50"]["f1"])
        row["precision_gap"] = abs(per_dataset["steps20"]["precision"] - per_dataset["steps50"]["precision"])
        row["recall_gap"] = abs(per_dataset["steps20"]["recall"] - per_dataset["steps50"]["recall"])
        row["f1_gap"] = abs(per_dataset["steps20"]["f1"] - per_dataset["steps50"]["f1"])
        row["steps20_meets_target"] = int(
            per_dataset["steps20"]["precision"] >= 0.90 and per_dataset["steps20"]["recall"] >= 0.50
        )
        row["steps50_meets_target"] = int(
            per_dataset["steps50"]["precision"] >= 0.90 and per_dataset["steps50"]["recall"] >= 0.50
        )
        row["both_meet_target"] = int(row["steps20_meets_target"] and row["steps50_meets_target"])
        rows.append(row)

    return rows


def add_reference_rules(rows: List[Dict[str, object]], datasets: Dict[str, Dict[str, object]]) -> None:
    reference_specs = [
        {
            "family": "reference_phase3_interaction",
            "rule_name": "reference|accepted_high_count|margin<0p03|count>=2",
            "trigger_fn": lambda block: block["ap_low_margin_high_kl_count_lt_0p03"] >= 2,
            "deployable": 1,
            "core_signal": "accepted_prefix_interaction",
        },
        {
            "family": "reference_phase3_interaction",
            "rule_name": "reference|accepted_drift_count|margin<0p03|count>=1",
            "trigger_fn": lambda block: block["ap_low_margin_drift_count_lt_0p03"] >= 1,
            "deployable": 1,
            "core_signal": "accepted_prefix_interaction",
        },
        {
            "family": "reference_phase3_source_or",
            "rule_name": "reference|accepted_high_count|margin<0p05|count>=2|or|bonus_ge0_count>=2",
            "trigger_fn": lambda block: (
                block["ap_low_margin_high_kl_count_lt_0p05"] >= 2
                or block["early_bonus_fragile_count_pos_ge_0"] >= 2
            ),
            "deployable": 1,
            "core_signal": "accepted_prefix_interaction_plus_optional_bonus",
        },
    ]
    rows.extend(evaluate_rule_specs(datasets, reference_specs))


def select_shortlist(rule_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    shortlisted: List[Dict[str, object]] = []
    seen = set()

    selections = [
        ("best_steps50_precision_at_recall50", lambda rows: sorted(
            [row for row in rows if row["steps50_recall"] >= 0.50],
            key=lambda row: (-row["steps50_precision"], -row["steps50_recall"], row["steps50_trigger_rate"]),
        )),
        ("best_steps50_f1", lambda rows: sorted(rows, key=lambda row: (-row["steps50_f1"], -row["steps50_precision"]))),
        ("best_robust_min_f1", lambda rows: sorted(rows, key=lambda row: (-row["min_f1"], -row["min_precision"]))),
        ("best_robust_min_precision", lambda rows: sorted(rows, key=lambda row: (-row["min_precision"], -row["min_recall"]))),
        ("best_accepted_only", lambda rows: sorted(
            [row for row in rows if row["core_signal"] == "accepted_prefix_interaction"],
            key=lambda row: (-row["steps50_f1"], -row["min_precision"]),
        )),
        ("best_with_bonus", lambda rows: sorted(
            [row for row in rows if row["core_signal"] == "accepted_prefix_interaction_plus_optional_bonus"],
            key=lambda row: (-row["steps50_f1"], -row["min_precision"]),
        )),
    ]

    for label, selector in selections:
        ranked = selector(rule_rows)
        if not ranked:
            continue
        row = dict(ranked[0])
        row["selection_label"] = label
        key = row["rule_name"]
        if key not in seen:
            shortlisted.append(row)
            seen.add(key)
        else:
            for existing in shortlisted:
                if existing["rule_name"] == key:
                    existing["selection_label"] = existing["selection_label"] + ";" + label
                    break

    return shortlisted


def write_csv(path: str, rows: List[Dict[str, object]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "steps20": build_dataset("steps20", args.steps20_csv),
        "steps50": build_dataset("steps50", args.steps50_csv),
    }

    block_feature_rows: List[Dict[str, object]] = []
    token_summary_rows: List[Dict[str, object]] = []
    for dataset in datasets.values():
        block_feature_rows.extend(dataset["block_features"])
        token_summary_rows.extend(dataset["token_summary"])

    rule_rows = evaluate_rule_specs(datasets, build_rule_specs())
    add_reference_rules(rule_rows, datasets)
    rule_rows.sort(key=lambda row: (-row["steps50_f1"], -row["min_precision"], -row["steps20_f1"]))

    shortlist_rows = select_shortlist(rule_rows)

    write_csv(
        os.path.join(output_dir, f"{args.output_prefix}_block_features.csv"),
        block_feature_rows,
    )
    write_csv(
        os.path.join(output_dir, f"{args.output_prefix}_token_summary.csv"),
        token_summary_rows,
    )
    write_csv(
        os.path.join(output_dir, f"{args.output_prefix}_rule_metrics.csv"),
        rule_rows,
    )
    write_csv(
        os.path.join(output_dir, f"{args.output_prefix}_shortlist.csv"),
        shortlist_rows,
    )

    for row in shortlist_rows:
        print(
            f"{row['selection_label']}: {row['rule_name']} | "
            f"steps20 p/r/f1={row['steps20_precision']:.3f}/{row['steps20_recall']:.3f}/{row['steps20_f1']:.3f} | "
            f"steps50 p/r/f1={row['steps50_precision']:.3f}/{row['steps50_recall']:.3f}/{row['steps50_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
