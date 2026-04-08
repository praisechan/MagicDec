import argparse
import csv
import json
import math
import os
import re
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from MagicDec.Data.data_converter import convert_pg19_dataset
from MagicDec.Engine.RetrievalAttention_new.backend import LMBackend
from MagicDec.Engine.utils import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile draft confidence margins and final-verify rejection histogram (2-stage only)."
    )
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3.1-8B")
    parser.add_argument("--dataset", type=str, default="pg19", choices=["pg19", "longbenchv1"])
    parser.add_argument("--task", type=str, default="gov_report")
    parser.add_argument("--prefix_len", type=int, default=8192)
    parser.add_argument("--gamma1", type=int, default=8)
    parser.add_argument("--budget1", type=float, default=0.05)
    parser.add_argument("--num_max_token", type=int, default=64)
    parser.add_argument("--num_eval_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="MagicDec/tests/RetrievalAttention_new/logs/margin_profile",
    )
    parser.add_argument("--bin_width", type=float, default=0.02)
    return parser.parse_args()


def sanitize_tag(value):
    text = str(value)
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]", "-", text)
    return text


def format_float_tag(value):
    txt = f"{value:.10g}"
    return sanitize_tag(txt.replace(".", "p"))


def load_longbench_config():
    base = os.path.join(PROJECT_ROOT, "Engine", "RetrievalAttention_new", "benchmark", "longbench", "config")
    with open(os.path.join(base, "model2path.json"), "r", encoding="utf-8") as f:
        model2path = json.load(f)
    with open(os.path.join(base, "model2maxlen.json"), "r", encoding="utf-8") as f:
        model2maxlen = json.load(f)
    with open(os.path.join(base, "dataset2prompt.json"), "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    return model2path, model2maxlen, dataset2prompt


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


def margin_to_bin_index(margin_value, bin_width, num_bins):
    clipped = min(1.0, max(0.0, float(margin_value)))
    raw = int((clipped - 0.0) / bin_width)
    return min(raw, num_bins - 1)


def build_histogram_rows(bin_width, drafted_counts, rejected_counts):
    rows = []
    for idx, drafted_count in enumerate(drafted_counts):
        left = idx * bin_width
        right = min(1.0, (idx + 1) * bin_width)
        center = (left + right) / 2.0
        rejected_count = rejected_counts[idx]
        reject_rate = (rejected_count / drafted_count) if drafted_count > 0 else 0.0
        rows.append(
            {
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
    headers = [
        "bin_index",
        "bin_left",
        "bin_right",
        "bin_center",
        "drafted_token_count",
        "rejected_token_count",
        "reject_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def load_pg19_dataset():
    return load_dataset("emozilla/pg19", split="test")

def get_pg19_prompt_format():
    return (
        "You are given a passage from a book. Read the passage carefully.\n\n"
        "Passage:\n{text}\n\n"
        "Now, continue the text naturally and coherently based on the passage above.\n\n"
        "Continuation:"
    )
  
def main():
    args = parse_args()
    if not (0.0 < args.bin_width <= 1.0):
        raise ValueError("bin_width must be in (0, 1].")

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
    if args.dataset == "pg19":
        prompt_format = get_pg19_prompt_format()
    else:
        prompt_format = dataset2prompt[args.task]

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

    num_bins = int(math.ceil(1.0 / args.bin_width))
    drafted_counts = [0 for _ in range(num_bins)]
    rejected_counts = [0 for _ in range(num_bins)]

    total_included_drafted_tokens = 0
    total_rejected_tokens = 0

    total_steps = min(args.num_eval_steps, len(dataset))

    for step in tqdm(range(total_steps), total=total_steps):
        batch = dataset[step]
        input_ids = engine.preprocess_input(
            batch,
            prompt_format,
            args.dataset,
            args.prefix_len,
        )
        attention_masks = engine.attention_masks

        # # For Qwen2.5-32B
        # if input_ids.shape[1] > 22000:
        #     continue

        # We reuse setup_caches for draft cache lifecycle consistency. Early-verify caches are not used.
        engine.setup_caches(
            input_ids=input_ids,
            attention_masks=attention_masks,
            budget1=args.budget1,
            budget2=args.budget1,
            budget2_high=args.budget1,
            estimate_ratio=0.25,
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
        while generated_tokens < args.num_max_token:
            remaining = args.num_max_token - generated_tokens
            if remaining <= 0:
                break

            draft_steps = min(args.gamma1, remaining)
            draft_snapshot = engine.snapshot_state("draft")
            final_snapshot = engine.snapshot_state("final_verify")

            drafted_tokens, drafted_margins, _ = engine.speculate_with_confidence(current_token, draft_steps)
            if drafted_tokens.numel() == 0:
                break

            eos_idx = first_eos_idx(drafted_tokens, eos_id)
            if eos_idx >= 0:
                drafted_tokens = drafted_tokens[:, : eos_idx + 1]
                drafted_margins = drafted_margins[:, : eos_idx + 1]

            final_outputs = engine.final_verify(final_current_token, drafted_tokens)
            compared_final = final_outputs[:, : drafted_tokens.shape[1]]
            mismatch = first_mismatch_idx(drafted_tokens, compared_final)

            if mismatch >= 0:
                included_len = mismatch + 1
                accepted_len = mismatch
            else:
                included_len = drafted_tokens.shape[1]
                accepted_len = drafted_tokens.shape[1]

            for idx in range(included_len):
                margin_value = float(drafted_margins[0, idx])
                bin_idx = margin_to_bin_index(margin_value, args.bin_width, num_bins)
                drafted_counts[bin_idx] += 1
                total_included_drafted_tokens += 1

                is_rejected = mismatch >= 0 and idx == mismatch
                if is_rejected:
                    rejected_counts[bin_idx] += 1
                    total_rejected_tokens += 1

            authoritative = torch.cat(
                [
                    drafted_tokens[:, :accepted_len],
                    final_outputs[:, accepted_len : accepted_len + 1],
                ],
                dim=1,
            )

            eos_authoritative = first_eos_idx(authoritative, eos_id)
            if eos_authoritative >= 0:
                authoritative = authoritative[:, : eos_authoritative + 1]

            authoritative = authoritative[:, :remaining]
            if authoritative.numel() == 0:
                break

            engine.revert_to("draft", draft_snapshot)
            engine.revert_to("final_verify", final_snapshot)

            authoritative_prefix = authoritative[:, :-1]
            engine.commit_prefix("final_verify", final_current_token, authoritative_prefix)
            engine.commit_prefix(
                "draft",
                current_token,
                authoritative_prefix,
                sync_source="final_verify",
                source_replayed=True,
            )

            current_token = authoritative[:, -1:]
            final_current_token = authoritative[:, -1:]
            generated_tokens += authoritative.shape[1]

            if int(current_token[0, 0]) == eos_id:
                break

        engine.clear_kv()

    global_reject_rate = (
        float(total_rejected_tokens) / float(total_included_drafted_tokens)
        if total_included_drafted_tokens > 0
        else 0.0
    )

    rows = build_histogram_rows(args.bin_width, drafted_counts, rejected_counts)

    output_name = (
        "margin_hist"
        f"_num_eval_step_{sanitize_tag(args.num_eval_steps)}"
        f"_num_max_token_{sanitize_tag(args.num_max_token)}"
        f"_model_name_{sanitize_tag(args.model_name)}"
        f"_dataset_{sanitize_tag(args.dataset)}"
        f"_task_{sanitize_tag(args.task)}"
        f"_budget1_{format_float_tag(args.budget1)}"
        f"_gamma1_{sanitize_tag(args.gamma1)}"
        ".csv"
    )
    output_csv_path = os.path.join(logs_dir, output_name)
    write_histogram_csv(output_csv_path, rows)

    print(f"total_included_drafted_tokens: {total_included_drafted_tokens}")
    print(f"total_rejected_tokens: {total_rejected_tokens}")
    print(f"global_reject_rate: {global_reject_rate:.10f}")
    print(f"output_csv: {output_csv_path}")


if __name__ == "__main__":
    main()
