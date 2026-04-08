import argparse
import csv
import os
from collections import defaultdict
from typing import Callable, Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Scan pre-final block-level rejection indicators on Phase 3 profiles.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="")
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


def block_rows_for_indicator(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if row["is_bonus_position"] == "0" and row["pre_final_features_available"] == "1"
    ]


def evaluate_rule(blocks, rule_name: str, predicate: Callable[[Dict[str, str]], bool]) -> Dict[str, object]:
    tp = fp = tn = fn = 0
    triggered_blocks = 0
    rejected_blocks = 0
    accepted_blocks = 0

    for key, rows in sorted(blocks.items()):
        block_rows = block_rows_for_indicator(rows)
        block_rejected = int(rows[0]["block_rejected"])
        triggered = any(predicate(row) for row in block_rows)
        if block_rejected:
            rejected_blocks += 1
        else:
            accepted_blocks += 1
        if triggered:
            triggered_blocks += 1
            if block_rejected:
                tp += 1
            else:
                fp += 1
        else:
            if block_rejected:
                fn += 1
            else:
                tn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    trigger_rate = safe_div(triggered_blocks, triggered_blocks + tn + fn)

    return {
        "rule_name": rule_name,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "trigger_rate": trigger_rate,
        "rejected_blocks": rejected_blocks,
        "accepted_blocks": accepted_blocks,
    }


def rules():
    return [
        (
            "early_margin_lt_0.10_and_early_entropy_gt_3.5",
            lambda r: to_float(r, "early_margin", 1.0) < 0.10 and to_float(r, "early_entropy", 0.0) > 3.5,
        ),
        (
            "early_margin_lt_0.10",
            lambda r: to_float(r, "early_margin", 1.0) < 0.10,
        ),
        (
            "early_entropy_gt_3.5",
            lambda r: to_float(r, "early_entropy", 0.0) > 3.5,
        ),
        (
            "early_margin_lt_0.15_and_early_entropy_gt_3.2",
            lambda r: to_float(r, "early_margin", 1.0) < 0.15 and to_float(r, "early_entropy", 0.0) > 3.2,
        ),
        (
            "early_margin_lt_0.10_and_top10_overlap_draft_early_le_8",
            lambda r: to_float(r, "early_margin", 1.0) < 0.10 and to_int(r, "top10_overlap_draft_early", 10) <= 8,
        ),
        (
            "early_margin_lt_0.10_and_kl_early_draft_gt_0.02",
            lambda r: to_float(r, "early_margin", 1.0) < 0.10 and to_float(r, "kl_early_draft", 0.0) > 0.02,
        ),
        (
            "early_margin_lt_0.03_and_kl_early_draft_gt_0.01",
            lambda r: to_float(r, "early_margin", 1.0) < 0.03 and to_float(r, "kl_early_draft", 0.0) > 0.01,
        ),
        (
            "early_margin_lt_0.03_and_kl_early_draft_gt_0.02",
            lambda r: to_float(r, "early_margin", 1.0) < 0.03 and to_float(r, "kl_early_draft", 0.0) > 0.02,
        ),
        (
            "early_margin_lt_0.10_and_source_kind_early_bonus",
            lambda r: to_float(r, "early_margin", 1.0) < 0.10 and r["source_kind"] == "early_bonus",
        ),
        (
            "early_margin_lt_0.10_and_verify_mode_high",
            lambda r: to_float(r, "early_margin", 1.0) < 0.10 and r["source_verify_mode"] == "high",
        ),
        (
            "draft_margin_missing_or_early_over_draft_entropy_delta_gt_0.5_and_early_margin_lt_0.15",
            lambda r: (
                to_float(r, "early_margin", 1.0) < 0.15
                and (
                    r["draft_entropy"] == ""
                    or (to_float(r, "early_entropy", 0.0) - to_float(r, "draft_entropy", 0.0)) > 0.5
                )
            ),
        ),
        (
            "source_kind_early_bonus_and_early_entropy_gt_3.3",
            lambda r: r["source_kind"] == "early_bonus" and to_float(r, "early_entropy", 0.0) > 3.3,
        ),
    ]


def main():
    args = parse_args()
    rows = read_rows(args.input_csv)
    blocks = build_blocks(rows)
    results = [evaluate_rule(blocks, name, fn) for name, fn in rules()]
    results.sort(key=lambda row: (-row["precision"], -row["recall"], row["trigger_rate"]))

    output_csv = args.output_csv or os.path.join(
        os.path.dirname(args.input_csv),
        f"phase3_indicator_scan_{os.path.basename(args.input_csv)}",
    )
    fieldnames = list(results[0].keys()) if results else ["rule_name"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        print(
            f"{row['rule_name']}: precision={row['precision']:.3f} recall={row['recall']:.3f} "
            f"f1={row['f1']:.3f} trigger_rate={row['trigger_rate']:.3f} tp={row['tp']} fp={row['fp']} fn={row['fn']}"
        )
    print(f"Wrote {len(results)} rows to {output_csv}")


if __name__ == "__main__":
    main()
