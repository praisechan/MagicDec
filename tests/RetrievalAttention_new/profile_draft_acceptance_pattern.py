import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from MagicDec.Engine.RetrievalAttention_new.backend import LMBackend
from MagicDec.Engine.utils import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile drafted-token positional matches against unforced full-attention "
            "final generation."
        )
    )
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3.1-8B")
    parser.add_argument("--dataset", type=str, default="pg19", choices=["pg19", "longbenchv1"])
    parser.add_argument("--task", type=str, default="gov_report")
    parser.add_argument("--prefix_len", type=int, default=8192)
    parser.add_argument("--gamma1", type=int, default=8)
    parser.add_argument("--budget1", type=float, default=0.05)
    parser.add_argument("--estimate_ratio", type=float, default=0.25)
    parser.add_argument("--num_max_token", type=int, default=64)
    parser.add_argument("--num_eval_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="tests/RetrievalAttention_new/logs/acceptance_pattern_profile",
    )
    parser.add_argument("--output_prefix", type=str, default="acceptance_pattern")
    parser.add_argument("--save_figure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--figure_format", type=str, default="png", choices=["png", "pdf"])
    return parser.parse_args()


def sanitize_tag(value):
    text = str(value).replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "-", text)


def format_float_tag(value):
    return sanitize_tag(f"{value:.10g}".replace(".", "p"))


def load_longbench_config():
    base = os.path.join(PROJECT_ROOT, "Engine", "RetrievalAttention_new", "benchmark", "longbench", "config")
    with open(os.path.join(base, "model2path.json"), "r", encoding="utf-8") as f:
        model2path = json.load(f)
    with open(os.path.join(base, "model2maxlen.json"), "r", encoding="utf-8") as f:
        model2maxlen = json.load(f)
    with open(os.path.join(base, "dataset2prompt.json"), "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    return model2path, model2maxlen, dataset2prompt


def load_pg19_dataset():
    return load_dataset("emozilla/pg19", split="test")


def get_pg19_prompt_format():
    return (
        "You are given a passage from a book. Read the passage carefully.\n\n"
        "Passage:\n{text}\n\n"
        "Now, continue the text naturally and coherently based on the passage above.\n\n"
        "Continuation:"
    )


def first_eos_idx(tokens, eos_id):
    for idx in range(tokens.shape[1]):
        if int(tokens[0, idx]) == eos_id:
            return idx
    return -1


def first_mismatch_idx(lhs, rhs):
    max_len = min(lhs.shape[1], rhs.shape[1])
    for idx in range(max_len):
        if int(lhs[0, idx]) != int(rhs[0, idx]):
            return idx
    return -1


def token_ids_to_text(ids):
    return " ".join(str(int(x)) for x in ids)


def build_output_stem(args):
    return (
        f"{sanitize_tag(args.output_prefix)}"
        f"_steps_{sanitize_tag(args.num_eval_steps)}"
        f"_max_token_{sanitize_tag(args.num_max_token)}"
        f"_model_{sanitize_tag(args.model_name)}"
        f"_dataset_{sanitize_tag(args.dataset)}"
        f"_task_{sanitize_tag(args.task)}"
        f"_prefix_{sanitize_tag(args.prefix_len)}"
        f"_budget1_{format_float_tag(args.budget1)}"
        f"_gamma1_{sanitize_tag(args.gamma1)}"
        f"_seed_{sanitize_tag(args.seed)}"
    )


def append_row(rows, args, row):
    merged = dict(vars(args))
    merged.update(row)
    rows.append(merged)


def write_csv(path, args, rows):
    arg_fields = list(vars(args).keys())
    metric_fields = [
        "row_type",
        "dataset_index",
        "draft_block_index",
        "generated_before_block",
        "gamma_denominator",
        "drafted_len",
        "final_len",
        "accepted_prefix_len",
        "full_match_count",
        "post_rejection_len",
        "post_rejection_match_count",
        "post_rejection_contiguous_match_count",
        "mismatch_index",
        "is_rejected_case",
        "acceptance_rate",
        "actual_match_rate",
        "rejected_case_acceptance_rate",
        "rejected_case_actual_match_rate",
        "match_pattern",
        "draft_token_ids",
        "final_token_ids",
        "num_blocks",
        "num_rejected_blocks",
        "total_gamma_denominator",
        "total_accepted_prefix_tokens",
        "total_full_match_tokens",
        "total_rejected_case_denominator",
        "total_rejected_case_accepted_prefix_tokens",
        "total_rejected_case_full_match_tokens",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=arg_fields + metric_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_rate(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def plot_figure(path, summary, rejection_position_stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "Accepted prefix",
        "Actual match",
        "Rejected prefix",
        "Rejected actual",
    ]
    values = [
        summary["acceptance_rate"],
        summary["actual_match_rate"],
        summary["rejected_case_acceptance_rate"],
        summary["rejected_case_actual_match_rate"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    axes[0].bar(labels, [100.0 * value for value in values], color=["#4C78A8", "#59A14F", "#F28E2B", "#E15759"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title("Draft Acceptance vs. Full Positional Match")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    xs = sorted(rejection_position_stats)
    if xs:
        avg_post_match = [
            safe_rate(
                rejection_position_stats[x]["post_rejection_match_count"],
                rejection_position_stats[x]["post_rejection_len"],
            )
            * 100.0
            for x in xs
        ]
        counts = [rejection_position_stats[x]["count"] for x in xs]
        axes[1].plot(xs, avg_post_match, marker="o", color="#4C78A8", label="Post-rejection match rate")
        axes_count = axes[1].twinx()
        axes_count.bar(xs, counts, alpha=0.25, color="#BAB0AC", label="Rejected blocks")
        axes[1].set_ylabel("Post-rejection match rate (%)")
        axes_count.set_ylabel("Rejected blocks")
    else:
        axes[1].text(0.5, 0.5, "No rejected draft blocks", ha="center", va="center")
        axes[1].set_yticks([])

    axes[1].set_xlabel("Rejection index in draft block")
    axes[1].set_title("Matches After the First Rejection")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.gamma1 <= 0:
        raise ValueError("gamma1 must be positive.")
    if args.num_max_token <= 0:
        raise ValueError("num_max_token must be positive.")

    setup_seed(args.seed)

    model2path, model2maxlen, dataset2prompt = load_longbench_config()
    if args.model_name not in model2path:
        raise KeyError(f"Unknown model_name: {args.model_name}")
    if args.task not in dataset2prompt:
        raise KeyError(f"Unknown task: {args.task}")

    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_device = "auto" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = model2path[args.model_name]
    max_length = model2maxlen[args.model_name]
    prompt_format = get_pg19_prompt_format() if args.dataset == "pg19" else dataset2prompt[args.task]

    engine = LMBackend(dtype=torch.bfloat16, device=runtime_device, dec_len=args.gamma1 + 1)
    engine.load_model(model_path, max_length, torch.bfloat16, model_device, 1)

    tokenizer = engine.model.tokenizer
    eos_id = tokenizer.eos_token_id

    if args.dataset == "pg19":
        dataset = load_pg19_dataset()
    else:
        dataset = load_dataset("THUDM/LongBench", args.task, split="test", trust_remote_code=True)

    logs_dir = args.logs_dir
    if not os.path.isabs(logs_dir):
        logs_dir = os.path.join(PROJECT_ROOT, logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    rows = []
    rejection_position_stats = defaultdict(
        lambda: {
            "count": 0,
            "post_rejection_len": 0,
            "post_rejection_match_count": 0,
        }
    )

    total_blocks = 0
    total_rejected_blocks = 0
    total_gamma_denominator = 0
    total_accepted_prefix_tokens = 0
    total_full_match_tokens = 0
    rejected_case_denominator = 0
    rejected_case_accepted_prefix_tokens = 0
    rejected_case_full_match_tokens = 0

    total_steps = min(args.num_eval_steps, len(dataset))

    for step in tqdm(range(total_steps), total=total_steps):
        batch = dataset[step]
        input_ids = engine.preprocess_input(batch, prompt_format, args.dataset, args.prefix_len)
        attention_masks = engine.attention_masks

        # For Qwen2.5-32B.
        if input_ids.shape[1] > 50000:
            continue

        engine.setup_caches(
            input_ids=input_ids,
            attention_masks=attention_masks,
            budget1=args.budget1,
            budget2=args.budget1,
            budget2_high=args.budget1,
            estimate_ratio=args.estimate_ratio,
            max_new_tokens=args.num_max_token + args.gamma1 + 8,
        )
        engine.setup_final_verify_cache(
            input_ids=input_ids,
            attention_masks=attention_masks,
            max_new_tokens=args.num_max_token + args.gamma1 + 8,
        )

        current_token = engine.encode(input_ids=input_ids)
        final_current_token = engine.prefill_tokens["final_verify"].clone()

        generated_tokens = 0
        block_index = 0
        while generated_tokens < args.num_max_token:
            remaining = args.num_max_token - generated_tokens
            draft_steps = min(args.gamma1, remaining)
            if draft_steps <= 0:
                break

            draft_snapshot = engine.snapshot_state("draft")

            drafted_tokens, _, _ = engine.speculate_with_confidence(current_token, draft_steps)
            if drafted_tokens.numel() == 0:
                break

            draft_eos_idx = first_eos_idx(drafted_tokens, eos_id)
            if draft_eos_idx >= 0:
                drafted_tokens = drafted_tokens[:, : draft_eos_idx + 1]

            final_tokens = engine.final_generate(final_current_token, drafted_tokens.shape[1])
            final_eos_idx = first_eos_idx(final_tokens, eos_id)
            if final_eos_idx >= 0:
                final_tokens = final_tokens[:, : final_eos_idx + 1]

            profile_len = min(drafted_tokens.shape[1], final_tokens.shape[1])
            if profile_len <= 0:
                break

            profiled_draft = drafted_tokens[:, :profile_len]
            profiled_final = final_tokens[:, :profile_len]

            mismatch_index = first_mismatch_idx(profiled_draft, profiled_final)
            is_rejected_case = mismatch_index >= 0
            accepted_prefix_len = mismatch_index if is_rejected_case else profile_len
            match_values = [
                int(profiled_draft[0, idx]) == int(profiled_final[0, idx])
                for idx in range(profile_len)
            ]
            full_match_count = sum(1 for matched in match_values if matched)

            post_rejection_len = 0
            post_rejection_match_count = 0
            post_rejection_contiguous_match_count = 0
            if is_rejected_case:
                post_rejection_len = max(profile_len - mismatch_index - 1, 0)
                post_rejection_match_count = sum(
                    1 for matched in match_values[mismatch_index + 1 :] if matched
                )
                for matched in match_values[mismatch_index + 1 :]:
                    if not matched:
                        break
                    post_rejection_contiguous_match_count += 1

                rejection_position_stats[mismatch_index]["count"] += 1
                rejection_position_stats[mismatch_index]["post_rejection_len"] += post_rejection_len
                rejection_position_stats[mismatch_index]["post_rejection_match_count"] += post_rejection_match_count

            gamma_denominator = profile_len
            acceptance_rate = safe_rate(accepted_prefix_len, gamma_denominator)
            actual_match_rate = safe_rate(full_match_count, gamma_denominator)
            rejected_case_acceptance_rate = acceptance_rate if is_rejected_case else ""
            rejected_case_actual_match_rate = actual_match_rate if is_rejected_case else ""

            append_row(
                rows,
                args,
                {
                    "row_type": "block",
                    "dataset_index": step,
                    "draft_block_index": block_index,
                    "generated_before_block": generated_tokens,
                    "gamma_denominator": gamma_denominator,
                    "drafted_len": int(drafted_tokens.shape[1]),
                    "final_len": int(final_tokens.shape[1]),
                    "accepted_prefix_len": accepted_prefix_len,
                    "full_match_count": full_match_count,
                    "post_rejection_len": post_rejection_len,
                    "post_rejection_match_count": post_rejection_match_count,
                    "post_rejection_contiguous_match_count": post_rejection_contiguous_match_count,
                    "mismatch_index": mismatch_index,
                    "is_rejected_case": int(is_rejected_case),
                    "acceptance_rate": f"{acceptance_rate:.10f}",
                    "actual_match_rate": f"{actual_match_rate:.10f}",
                    "rejected_case_acceptance_rate": (
                        f"{rejected_case_acceptance_rate:.10f}" if is_rejected_case else ""
                    ),
                    "rejected_case_actual_match_rate": (
                        f"{rejected_case_actual_match_rate:.10f}" if is_rejected_case else ""
                    ),
                    "match_pattern": "".join("1" if matched else "0" for matched in match_values),
                    "draft_token_ids": token_ids_to_text(profiled_draft[0]),
                    "final_token_ids": token_ids_to_text(profiled_final[0]),
                    "num_blocks": "",
                    "num_rejected_blocks": "",
                    "total_gamma_denominator": "",
                    "total_accepted_prefix_tokens": "",
                    "total_full_match_tokens": "",
                    "total_rejected_case_denominator": "",
                    "total_rejected_case_accepted_prefix_tokens": "",
                    "total_rejected_case_full_match_tokens": "",
                },
            )

            total_blocks += 1
            total_gamma_denominator += gamma_denominator
            total_accepted_prefix_tokens += accepted_prefix_len
            total_full_match_tokens += full_match_count
            if is_rejected_case:
                total_rejected_blocks += 1
                rejected_case_denominator += gamma_denominator
                rejected_case_accepted_prefix_tokens += accepted_prefix_len
                rejected_case_full_match_tokens += full_match_count

            authoritative = final_tokens[:, :profile_len]
            generated_tokens += authoritative.shape[1]
            block_index += 1

            if first_eos_idx(authoritative, eos_id) >= 0:
                break

            engine.revert_to("draft", draft_snapshot)
            authoritative_prefix = authoritative[:, :-1]
            engine.commit_prefix(
                "draft",
                current_token,
                authoritative_prefix,
                sync_source="final_verify",
                source_replayed=True,
            )

            current_token = authoritative[:, -1:]
            final_current_token = authoritative[:, -1:]

        engine.clear_kv()

    summary = {
        "acceptance_rate": safe_rate(total_accepted_prefix_tokens, total_gamma_denominator),
        "actual_match_rate": safe_rate(total_full_match_tokens, total_gamma_denominator),
        "rejected_case_acceptance_rate": safe_rate(
            rejected_case_accepted_prefix_tokens,
            rejected_case_denominator,
        ),
        "rejected_case_actual_match_rate": safe_rate(
            rejected_case_full_match_tokens,
            rejected_case_denominator,
        ),
    }

    append_row(
        rows,
        args,
        {
            "row_type": "summary",
            "dataset_index": "",
            "draft_block_index": "",
            "generated_before_block": "",
            "gamma_denominator": "",
            "drafted_len": "",
            "final_len": "",
            "accepted_prefix_len": "",
            "full_match_count": "",
            "post_rejection_len": "",
            "post_rejection_match_count": "",
            "post_rejection_contiguous_match_count": "",
            "mismatch_index": "",
            "is_rejected_case": "",
            "acceptance_rate": f"{summary['acceptance_rate']:.10f}",
            "actual_match_rate": f"{summary['actual_match_rate']:.10f}",
            "rejected_case_acceptance_rate": f"{summary['rejected_case_acceptance_rate']:.10f}",
            "rejected_case_actual_match_rate": f"{summary['rejected_case_actual_match_rate']:.10f}",
            "match_pattern": "",
            "draft_token_ids": "",
            "final_token_ids": "",
            "num_blocks": total_blocks,
            "num_rejected_blocks": total_rejected_blocks,
            "total_gamma_denominator": total_gamma_denominator,
            "total_accepted_prefix_tokens": total_accepted_prefix_tokens,
            "total_full_match_tokens": total_full_match_tokens,
            "total_rejected_case_denominator": rejected_case_denominator,
            "total_rejected_case_accepted_prefix_tokens": rejected_case_accepted_prefix_tokens,
            "total_rejected_case_full_match_tokens": rejected_case_full_match_tokens,
        },
    )

    stem = build_output_stem(args)
    output_csv_path = os.path.join(logs_dir, f"{stem}.csv")
    write_csv(output_csv_path, args, rows)

    output_figure_path = ""
    if args.save_figure:
        output_figure_path = os.path.join(logs_dir, f"{stem}.{args.figure_format}")
        plot_figure(output_figure_path, summary, rejection_position_stats)

    print(f"num_blocks: {total_blocks}")
    print(f"num_rejected_blocks: {total_rejected_blocks}")
    print(f"acceptance_rate: {summary['acceptance_rate']:.10f}")
    print(f"actual_match_rate: {summary['actual_match_rate']:.10f}")
    print(f"rejected_case_acceptance_rate: {summary['rejected_case_acceptance_rate']:.10f}")
    print(f"rejected_case_actual_match_rate: {summary['rejected_case_actual_match_rate']:.10f}")
    print(f"output_csv: {output_csv_path}")
    if output_figure_path:
        print(f"output_figure: {output_figure_path}")


if __name__ == "__main__":
    main()
