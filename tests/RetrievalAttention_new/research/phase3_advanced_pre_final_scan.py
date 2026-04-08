import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search stronger source-aware pre-final block rejection rules on Phase 3 profiles."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--details_csv", type=str, default="")
    parser.add_argument("--top_k", type=int, default=200)
    return parser.parse_args()


def read_rows(path: str) -> List[Dict[str, str]]:
    csv.field_size_limit(1 << 30)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: Dict[str, str], key: str, default=None):
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def to_int(row: Dict[str, str], key: str, default=None):
    value = row.get(key, "")
    if value == "":
        return default
    return int(value)


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def block_key(row: Dict[str, str]) -> Tuple[int, int]:
    return int(row["step"]), int(row["cycle_idx"])


def build_blocks(rows: List[Dict[str, str]]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[block_key(row)].append(row)
    return grouped


def indicator_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if row["is_bonus_position"] == "0" and row["pre_final_features_available"] == "1"
    ]


def build_block_cache(blocks):
    cache = []
    for key, rows in sorted(blocks.items()):
        usable_rows = indicator_rows(rows)
        accepted_prefix_rows = [row for row in usable_rows if row["source_kind"] == "accepted_prefix"]
        early_bonus_rows = [row for row in usable_rows if row["source_kind"] == "early_bonus"]
        cache.append(
            {
                "block_key": key,
                "rows": usable_rows,
                "accepted_prefix_rows": accepted_prefix_rows,
                "early_bonus_rows": early_bonus_rows,
                "block_rejected": int(rows[0]["block_rejected"]),
            }
        )
    return cache


def evaluate_rule(block_cache, rule_name: str, family: str, params: Dict[str, object], predicate):
    tp = fp = tn = fn = 0
    details = []

    for block in block_cache:
        triggered, reason = predicate(block)
        rejected = block["block_rejected"]
        if triggered and rejected:
            tp += 1
        elif triggered and not rejected:
            fp += 1
        elif rejected:
            fn += 1
        else:
            tn += 1
        details.append(
            {
                "rule_name": rule_name,
                "family": family,
                "step": block["block_key"][0],
                "cycle_idx": block["block_key"][1],
                "block_rejected": rejected,
                "triggered": int(triggered),
                "trigger_reason": reason,
            }
        )

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    trigger_rate = safe_div(tp + fp, tp + fp + tn + fn)

    result = {
        "rule_name": rule_name,
        "family": family,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "trigger_rate": trigger_rate,
    }
    for key, value in params.items():
        result[key] = value
    return result, details


def count_matching(rows: List[Dict[str, str]], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def build_rule_specs():
    specs = []

    accepted_margin_thresholds = [0.03, 0.05, 0.07, 0.10]
    accepted_kl_thresholds = [0.02, 0.03, 0.05]
    accepted_overlap_thresholds = [7, 8, 9]
    accepted_counts = [1, 2, 3]

    bonus_pos_thresholds = [0, 4, 8, 12]
    bonus_margin_thresholds = [0.02, 0.03, 0.05, 0.10]
    bonus_entropy_thresholds = [3.3, 3.5, 3.8, 4.0]
    bonus_counts = [1, 2, 3]

    for margin in accepted_margin_thresholds:
        for kl in accepted_kl_thresholds:
            for overlap in accepted_overlap_thresholds:
                for count in accepted_counts:
                    specs.append(
                        {
                            "family": "accepted_prefix_count",
                            "params": {
                                "accepted_margin_lt": margin,
                                "accepted_kl_gt": kl,
                                "accepted_overlap_le": overlap,
                                "accepted_count_ge": count,
                            },
                        }
                    )

    for pos in bonus_pos_thresholds:
        for margin in bonus_margin_thresholds:
            for entropy in bonus_entropy_thresholds:
                for count in bonus_counts:
                    specs.append(
                        {
                            "family": "early_bonus_count",
                            "params": {
                                "bonus_position_ge": pos,
                                "bonus_margin_lt": margin,
                                "bonus_entropy_gt": entropy,
                                "bonus_count_ge": count,
                            },
                        }
                    )

    for accepted_margin in accepted_margin_thresholds:
        for accepted_kl in accepted_kl_thresholds:
            for accepted_overlap in accepted_overlap_thresholds:
                for accepted_count in [1, 2]:
                    for bonus_pos in bonus_pos_thresholds:
                        for bonus_margin in bonus_margin_thresholds:
                            for bonus_entropy in bonus_entropy_thresholds:
                                for bonus_count in [1, 2]:
                                    specs.append(
                                        {
                                            "family": "piecewise_source_or",
                                            "params": {
                                                "accepted_margin_lt": accepted_margin,
                                                "accepted_kl_gt": accepted_kl,
                                                "accepted_overlap_le": accepted_overlap,
                                                "accepted_count_ge": accepted_count,
                                                "bonus_position_ge": bonus_pos,
                                                "bonus_margin_lt": bonus_margin,
                                                "bonus_entropy_gt": bonus_entropy,
                                                "bonus_count_ge": bonus_count,
                                            },
                                        }
                                    )

    for accepted_margin in accepted_margin_thresholds:
        for accepted_kl in accepted_kl_thresholds:
            for bonus_pos in bonus_pos_thresholds:
                for bonus_margin in bonus_margin_thresholds:
                    for bonus_entropy in bonus_entropy_thresholds:
                        for total_count in [2, 3, 4]:
                            specs.append(
                                {
                                    "family": "piecewise_sum_count",
                                    "params": {
                                        "accepted_margin_lt": accepted_margin,
                                        "accepted_kl_gt": accepted_kl,
                                        "bonus_position_ge": bonus_pos,
                                        "bonus_margin_lt": bonus_margin,
                                        "bonus_entropy_gt": bonus_entropy,
                                        "total_count_ge": total_count,
                                    },
                                }
                            )

    return specs


def make_rule(spec):
    family = spec["family"]
    params = spec["params"]

    if family == "accepted_prefix_count":
        def predicate(block):
            match_count = count_matching(
                block["accepted_prefix_rows"],
                lambda row: (
                    to_float(row, "early_margin", 1.0) < params["accepted_margin_lt"]
                    and (
                        to_float(row, "kl_early_draft", 0.0) > params["accepted_kl_gt"]
                        or to_int(row, "top10_overlap_draft_early", 10) <= params["accepted_overlap_le"]
                    )
                ),
            )
            return match_count >= params["accepted_count_ge"], f"accepted_prefix_count={match_count}"

    elif family == "early_bonus_count":
        def predicate(block):
            match_count = count_matching(
                block["early_bonus_rows"],
                lambda row: (
                    to_int(row, "position", 0) >= params["bonus_position_ge"]
                    and to_float(row, "early_margin", 1.0) < params["bonus_margin_lt"]
                    and to_float(row, "early_entropy", 0.0) > params["bonus_entropy_gt"]
                ),
            )
            return match_count >= params["bonus_count_ge"], f"early_bonus_count={match_count}"

    elif family == "piecewise_source_or":
        def predicate(block):
            accepted_count = count_matching(
                block["accepted_prefix_rows"],
                lambda row: (
                    to_float(row, "early_margin", 1.0) < params["accepted_margin_lt"]
                    and (
                        to_float(row, "kl_early_draft", 0.0) > params["accepted_kl_gt"]
                        or to_int(row, "top10_overlap_draft_early", 10) <= params["accepted_overlap_le"]
                    )
                ),
            )
            bonus_count = count_matching(
                block["early_bonus_rows"],
                lambda row: (
                    to_int(row, "position", 0) >= params["bonus_position_ge"]
                    and to_float(row, "early_margin", 1.0) < params["bonus_margin_lt"]
                    and to_float(row, "early_entropy", 0.0) > params["bonus_entropy_gt"]
                ),
            )
            triggered = (
                accepted_count >= params["accepted_count_ge"]
                or bonus_count >= params["bonus_count_ge"]
            )
            return triggered, f"accepted_prefix_count={accepted_count};early_bonus_count={bonus_count}"

    elif family == "piecewise_sum_count":
        def predicate(block):
            accepted_count = count_matching(
                block["accepted_prefix_rows"],
                lambda row: (
                    to_float(row, "early_margin", 1.0) < params["accepted_margin_lt"]
                    and to_float(row, "kl_early_draft", 0.0) > params["accepted_kl_gt"]
                ),
            )
            bonus_count = count_matching(
                block["early_bonus_rows"],
                lambda row: (
                    to_int(row, "position", 0) >= params["bonus_position_ge"]
                    and to_float(row, "early_margin", 1.0) < params["bonus_margin_lt"]
                    and to_float(row, "early_entropy", 0.0) > params["bonus_entropy_gt"]
                ),
            )
            total_count = accepted_count + bonus_count
            return total_count >= params["total_count_ge"], (
                f"accepted_prefix_count={accepted_count};early_bonus_count={bonus_count};total_count={total_count}"
            )

    else:
        raise ValueError(f"Unknown rule family: {family}")

    return predicate


def rule_name(spec) -> str:
    family = spec["family"]
    parts = [family]
    for key, value in spec["params"].items():
        parts.append(f"{key}={value}")
    return "|".join(parts)


def main():
    args = parse_args()
    rows = read_rows(args.input_csv)
    blocks = build_blocks(rows)
    block_cache = build_block_cache(blocks)

    results = []
    detail_rows = []
    for spec in build_rule_specs():
        name = rule_name(spec)
        predicate = make_rule(spec)
        result, details = evaluate_rule(block_cache, name, spec["family"], spec["params"], predicate)
        results.append(result)
        detail_rows.extend(details)

    results.sort(
        key=lambda row: (
            -(row["precision"] if row["recall"] >= 0.5 else -1.0),
            -row["recall"],
            -row["f1"],
            row["fp"],
            row["fn"],
            row["family"],
        )
    )

    output_csv = args.output_csv or os.path.join(
        os.path.dirname(args.input_csv),
        f"phase3_advanced_indicator_scan_{os.path.basename(args.input_csv)}",
    )
    details_csv = args.details_csv or os.path.join(
        os.path.dirname(output_csv),
        f"{os.path.splitext(os.path.basename(output_csv))[0]}_details.csv",
    )

    kept_results = results[: args.top_k]
    result_fieldnames = sorted({key for row in kept_results for key in row.keys()})
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_fieldnames)
        writer.writeheader()
        writer.writerows(kept_results)

    kept_rule_names = {row["rule_name"] for row in kept_results[:10]}
    kept_details = [row for row in detail_rows if row["rule_name"] in kept_rule_names]
    detail_fieldnames = list(kept_details[0].keys()) if kept_details else ["rule_name"]
    with open(details_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        writer.writerows(kept_details)

    for row in kept_results[:20]:
        print(
            f"{row['rule_name']}: precision={row['precision']:.3f} recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f} tp={row['tp']} fp={row['fp']} fn={row['fn']}"
        )
    print(f"Wrote {len(kept_results)} rows to {output_csv}")
    print(f"Wrote {len(kept_details)} detail rows to {details_csv}")


if __name__ == "__main__":
    main()
