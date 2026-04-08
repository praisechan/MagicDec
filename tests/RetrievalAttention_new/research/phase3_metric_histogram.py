import argparse
import csv
import math
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build token-distribution / reject-rate histograms for a Phase 3 metric column."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--bin_width", type=float, required=True)
    parser.add_argument("--metric_min", type=float, default=0.0)
    parser.add_argument("--metric_max", type=float, default=None)
    parser.add_argument("--source_kind", type=str, default="")
    parser.add_argument("--output_csv", type=str, default="")
    return parser.parse_args()


def read_rows(path):
    csv.field_size_limit(1 << 30)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value == "":
        return None
    return float(value)


def metric_to_bin_index(value, metric_min, bin_width, num_bins):
    clipped = min(metric_min + num_bins * bin_width, max(metric_min, float(value)))
    raw = int((clipped - metric_min) / bin_width)
    return min(raw, num_bins - 1)


def build_histogram_rows(metric_name, bin_width, metric_min, drafted_counts, rejected_counts):
    rows = []
    for idx, drafted_count in enumerate(drafted_counts):
        left = metric_min + idx * bin_width
        right = metric_min + (idx + 1) * bin_width
        center = (left + right) / 2.0
        rejected_count = rejected_counts[idx]
        reject_rate = (rejected_count / drafted_count) if drafted_count > 0 else 0.0
        rows.append(
            {
                "metric_name": metric_name,
                "bin_index": idx,
                "bin_left": f"{left:.6f}",
                "bin_right": f"{right:.6f}",
                "bin_center": f"{center:.6f}",
                "drafted_token_count": int(drafted_count),
                "rejected_token_count": int(rejected_count),
                "reject_rate": f"{reject_rate:.10f}",
            }
        )
    return rows


def write_histogram_csv(path, rows):
    fieldnames = [
        "metric_name",
        "bin_index",
        "bin_left",
        "bin_right",
        "bin_center",
        "drafted_token_count",
        "rejected_token_count",
        "reject_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.bin_width <= 0.0:
        raise ValueError("bin_width must be positive.")

    rows = read_rows(args.input_csv)
    filtered = [
        row
        for row in rows
        if row["is_bonus_position"] == "0"
        and row["pre_final_features_available"] == "1"
        and (not args.source_kind or row["source_kind"] == args.source_kind)
        and row.get(args.metric, "") != ""
    ]
    if not filtered:
        raise ValueError("No rows remain after applying the requested filters.")

    metric_values = [to_float(row[args.metric]) for row in filtered]
    observed_min = min(metric_values)
    observed_max = max(metric_values)
    metric_min = args.metric_min
    metric_max = args.metric_max if args.metric_max is not None else observed_max
    if metric_max <= metric_min:
        raise ValueError("metric_max must be greater than metric_min.")

    num_bins = int(math.ceil((metric_max - metric_min) / args.bin_width))
    drafted_counts = [0 for _ in range(num_bins)]
    rejected_counts = [0 for _ in range(num_bins)]

    total_drafted = 0
    total_rejected = 0
    for row in filtered:
        metric_value = to_float(row[args.metric])
        if metric_value is None:
            continue
        bin_idx = metric_to_bin_index(metric_value, metric_min, args.bin_width, num_bins)
        drafted_counts[bin_idx] += 1
        total_drafted += 1
        if row["is_rejected_position"] == "1":
            rejected_counts[bin_idx] += 1
            total_rejected += 1

    histogram_rows = build_histogram_rows(args.metric, args.bin_width, metric_min, drafted_counts, rejected_counts)

    output_csv = args.output_csv
    if not output_csv:
        source_suffix = f"_source_{args.source_kind}" if args.source_kind else ""
        metric_tag = args.metric.replace(".", "_")
        output_csv = os.path.join(
            os.path.dirname(args.input_csv),
            f"phase3_hist_{metric_tag}{source_suffix}_{os.path.basename(args.input_csv)}",
        )

    write_histogram_csv(output_csv, histogram_rows)

    global_reject_rate = (total_rejected / total_drafted) if total_drafted > 0 else 0.0
    print(f"metric: {args.metric}")
    print(f"source_kind: {args.source_kind or 'ALL'}")
    print(f"observed_metric_min: {observed_min:.10f}")
    print(f"observed_metric_max: {observed_max:.10f}")
    print(f"total_metric_rows: {total_drafted}")
    print(f"total_rejected_rows: {total_rejected}")
    print(f"global_reject_rate: {global_reject_rate:.10f}")
    print(f"output_csv: {output_csv}")


if __name__ == "__main__":
    main()
