import argparse
import csv
import json
import os
import sys
from datetime import datetime

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
from MagicDec.tests.RetrievalAttention_new.riskgate_utils import (
    build_verified_commit,
    first_eos_idx,
    first_mismatch_idx,
    format_float_tag,
    get_pg19_prompt_format,
    load_longbench_config,
    load_pg19_dataset,
    sanitize_tag,
    summarize_draft_features,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile final-stage reject risk using normal early-verify plus immediate final-verify labels."
    )
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3.1-8B")
    parser.add_argument("--dataset", type=str, default="pg19", choices=["pg19", "longbenchv1"])
    parser.add_argument("--task", type=str, default="gov_report")
    parser.add_argument("--prefix_len", type=int, default=32768)
    parser.add_argument("--gamma1", type=int, default=6)
    parser.add_argument("--budget1", type=float, default=0.02)
    parser.add_argument("--budget2", type=float, default=0.10)
    parser.add_argument("--budget2_high", type=float, default=0.25)
    parser.add_argument("--estimate_ratio", type=float, default=0.25)
    parser.add_argument("--num_max_token", type=int, default=100)
    parser.add_argument("--num_eval_steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="tests/RetrievalAttention_new/logs/final_reject_risk_profile",
    )
    return parser.parse_args()


def truncate_feature_dict(feature_dict, length):
    return {name: tensor[:, :length] for name, tensor in feature_dict.items()}


def build_run_dir(base_dir, args):
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(PROJECT_ROOT, base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"profile_{sanitize_tag(args.dataset)}_task_{sanitize_tag(args.task)}"
        f"_steps_{sanitize_tag(args.num_eval_steps)}"
        f"_prefix_{sanitize_tag(args.prefix_len)}"
        f"_tokens_{sanitize_tag(args.num_max_token)}"
        f"_g1_{sanitize_tag(args.gamma1)}"
        f"_b1_{format_float_tag(args.budget1)}"
        f"_b2_{format_float_tag(args.budget2)}"
        f"_{timestamp}"
    )
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def write_csv(path, rows):
    headers = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not headers:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def finalize_block_and_token_rows(block_row_base, token_row_bases, normal_accept_len, final_mismatch, committed_len):
    block_reject = int(final_mismatch >= 0 and final_mismatch < committed_len)
    reject_kind = "accepted"
    reject_offset = -1
    reject_draft_idx = -1
    if block_reject:
        reject_offset = final_mismatch
        if final_mismatch < normal_accept_len:
            reject_kind = "draft_token"
            reject_draft_idx = final_mismatch
        else:
            reject_kind = "early_bonus"

    block_row = {
        **block_row_base,
        "block_effective_len_in_final_eval": committed_len,
        "block_label_valid": 1,
        "block_final_reject": block_reject,
        "block_final_reject_kind": reject_kind,
        "block_final_reject_offset_in_block": reject_offset,
        "block_final_reject_draft_idx": reject_draft_idx,
        "chunk_final_mismatch_pos": final_mismatch,
        "chunk_final_accept_len": committed_len if final_mismatch < 0 else final_mismatch,
    }

    token_rows = []
    token_limit = min(len(token_row_bases), normal_accept_len, committed_len)
    for token_idx, token_row_base in enumerate(token_row_bases):
        token_valid = 0
        token_reject = 0
        if token_idx < token_limit:
            if reject_kind == "draft_token":
                if token_idx < reject_draft_idx:
                    token_valid = 1
                elif token_idx == reject_draft_idx:
                    token_valid = 1
                    token_reject = 1
            else:
                token_valid = 1

        token_rows.append(
            {
                **token_row_base,
                "token_label_valid": token_valid,
                "token_final_reject": token_reject,
                "block_label_valid": 1,
                "block_final_reject": block_reject,
                "block_final_reject_kind": reject_kind,
                "chunk_final_mismatch_pos": final_mismatch,
            }
        )

    return block_row, token_rows


def main():
    args = parse_args()
    setup_seed(args.seed)

    model2path, model2maxlen, dataset2prompt = load_longbench_config()
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_device = "auto" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = model2path[args.model_name]
    max_length = model2maxlen[args.model_name]
    if args.dataset == "pg19":
        prompt_format = get_pg19_prompt_format()
        dataset = load_pg19_dataset()
    else:
        prompt_format = dataset2prompt[args.task]
        dataset = load_dataset("THUDM/LongBench", args.task, split="test", trust_remote_code=True)

    engine = LMBackend(dtype=torch.bfloat16, device=runtime_device, dec_len=args.gamma1 + 1)
    engine.load_model(model_path, max_length, torch.bfloat16, model_device, 1)
    eos_id = engine.model.tokenizer.eos_token_id

    run_dir = build_run_dir(args.logs_dir, args)
    block_rows = []
    token_rows = []

    summary = {
        "profile_mode": "immediate_final",
        "total_steps": min(args.num_eval_steps, len(dataset)),
        "draft_decode_calls": 0,
        "early_normal_decode_calls": 0,
        "final_decode_calls": 0,
        "total_blocks": 0,
        "valid_blocks": 0,
        "rejected_blocks": 0,
        "bonus_rejected_blocks": 0,
        "valid_tokens": 0,
        "rejected_tokens": 0,
    }

    global_block_id = 0

    for step in tqdm(range(summary["total_steps"]), total=summary["total_steps"]):
        batch = dataset[step]
        input_ids = engine.preprocess_input(batch, prompt_format, args.dataset, args.prefix_len)
        attention_masks = engine.attention_masks

        engine.setup_caches(
            input_ids=input_ids,
            attention_masks=attention_masks,
            budget1=args.budget1,
            budget2=args.budget2,
            budget2_high=args.budget2_high,
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
        block_idx = 0

        while generated_tokens < args.num_max_token:
            remaining = args.num_max_token - generated_tokens
            draft_steps = min(args.gamma1, remaining)
            if draft_steps <= 0:
                break

            draft_snapshot = engine.snapshot_state("draft")
            early_snapshot = engine.snapshot_state("early_verify")
            early_high_snapshot = engine.snapshot_state("early_verify_high")
            final_snapshot = engine.snapshot_state("final_verify")
            start_token = current_token.clone()
            start_final_token = final_current_token.clone()

            draft_tokens, feature_dict = engine.speculate_with_features(current_token, draft_steps)
            if draft_tokens.numel() == 0:
                break
            summary["draft_decode_calls"] += draft_tokens.shape[1]

            eos_idx = first_eos_idx(draft_tokens, eos_id)
            if eos_idx >= 0:
                draft_tokens = draft_tokens[:, : eos_idx + 1]
                feature_dict = truncate_feature_dict(feature_dict, eos_idx + 1)

            early_outputs = engine.early_verify(current_token, draft_tokens, mode="normal")
            summary["early_normal_decode_calls"] += draft_tokens.shape[1] + 1

            normal_accept_len, normal_rejected_len, committed_online = build_verified_commit(
                draft_tokens,
                early_outputs,
                eos_id,
            )
            normal_mismatch_pos = first_mismatch_idx(draft_tokens, early_outputs[:, : draft_tokens.shape[1]])
            if normal_mismatch_pos < 0 and normal_accept_len < draft_tokens.shape[1]:
                normal_mismatch_pos = normal_accept_len

            final_outputs = engine.final_verify(final_current_token, committed_online)
            summary["final_decode_calls"] += committed_online.shape[1] + 1
            final_mismatch = first_mismatch_idx(committed_online, final_outputs[:, : committed_online.shape[1]])
            final_accept_len = committed_online.shape[1] if final_mismatch < 0 else final_mismatch

            authoritative_tokens = torch.cat(
                [
                    committed_online[:, :final_accept_len],
                    final_outputs[:, final_accept_len : final_accept_len + 1],
                ],
                dim=1,
            )
            authoritative_tokens = authoritative_tokens[:, :remaining]
            eos_authoritative = first_eos_idx(authoritative_tokens, eos_id)
            if eos_authoritative >= 0:
                authoritative_tokens = authoritative_tokens[:, : eos_authoritative + 1]
            if authoritative_tokens.numel() == 0:
                break

            feature_summary = summarize_draft_features(feature_dict)
            draft_token_ids = draft_tokens[0].detach().cpu().tolist()
            block_id = global_block_id
            global_block_id += 1

            block_row_base = {
                "step": step,
                "chunk_id": block_idx,
                "block_id": block_id,
                "draft_block_len": int(draft_tokens.shape[1]),
                "committed_block_len": int(committed_online.shape[1]),
                "block_start_in_chunk": 0,
                "block_end_in_chunk": int(committed_online.shape[1]),
                "normal_accept_len": int(normal_accept_len),
                "normal_rejected_len": int(normal_rejected_len),
                "normal_accept_ratio": float(normal_accept_len / max(draft_tokens.shape[1], 1)),
                "normal_mismatch_pos": int(normal_mismatch_pos),
                "normal_mismatch_pos_norm": (
                    float(normal_mismatch_pos / max(draft_tokens.shape[1] - 1, 1))
                    if normal_mismatch_pos >= 0
                    else -1.0
                ),
                "normal_bonus_token_id": int(early_outputs[0, normal_accept_len]),
                **{key: value for key, value in feature_summary.items() if not key.endswith("_values")},
            }

            token_row_bases = []
            for token_idx, token_id in enumerate(draft_token_ids):
                token_row_bases.append(
                    {
                        "step": step,
                        "chunk_id": block_idx,
                        "block_id": block_id,
                        "token_idx_in_block": token_idx,
                        "token_position_norm": (
                            float(token_idx / max(len(draft_token_ids) - 1, 1))
                            if len(draft_token_ids) > 1
                            else 0.0
                        ),
                        "draft_token_id": int(token_id),
                        "draft_token_reaches_normal_final_path": int(token_idx < normal_accept_len),
                        "top1_prob": feature_summary["top1_prob_values"][token_idx],
                        "top2_prob": feature_summary["top2_prob_values"][token_idx],
                        "margin": feature_summary["margin_values"][token_idx],
                        "entropy": feature_summary["entropy_values"][token_idx],
                        "is_min_margin_token": int(token_idx == feature_summary["block_min_margin_pos"]),
                    }
                )

            block_row, finalized_token_rows = finalize_block_and_token_rows(
                block_row_base,
                token_row_bases,
                normal_accept_len,
                final_mismatch,
                committed_online.shape[1],
            )
            block_rows.append(block_row)
            token_rows.extend(finalized_token_rows)
            summary["total_blocks"] += 1
            summary["valid_blocks"] += 1
            summary["rejected_blocks"] += int(block_row["block_final_reject"])
            summary["bonus_rejected_blocks"] += int(block_row["block_final_reject_kind"] == "early_bonus")
            for token_row in finalized_token_rows:
                summary["valid_tokens"] += int(token_row["token_label_valid"])
                summary["rejected_tokens"] += int(token_row["token_final_reject"])

            engine.revert_to("draft", draft_snapshot)
            engine.revert_to("early_verify", early_snapshot)
            engine.revert_to("early_verify_high", early_high_snapshot)
            engine.revert_to("final_verify", final_snapshot)

            final_prefix = authoritative_tokens[:, :-1]
            engine.commit_prefix("final_verify", start_final_token, final_prefix)
            engine.commit_prefix(
                "early_verify",
                start_token,
                final_prefix,
                sync_source="final_verify",
                source_replayed=True,
            )
            engine.commit_prefix(
                "early_verify_high",
                start_token,
                final_prefix,
                sync_source="final_verify",
                source_replayed=True,
            )
            engine.commit_prefix(
                "draft",
                start_token,
                final_prefix,
                sync_source="final_verify",
                source_replayed=True,
            )

            current_token = authoritative_tokens[:, -1:]
            final_current_token = authoritative_tokens[:, -1:]
            generated_tokens += authoritative_tokens.shape[1]
            block_idx += 1

            if int(current_token[0, 0]) == eos_id:
                break

        engine.clear_kv()

    summary["block_reject_rate"] = (
        float(summary["rejected_blocks"]) / float(summary["valid_blocks"])
        if summary["valid_blocks"] > 0
        else 0.0
    )
    summary["token_reject_rate"] = (
        float(summary["rejected_tokens"]) / float(summary["valid_tokens"])
        if summary["valid_tokens"] > 0
        else 0.0
    )

    block_csv = os.path.join(run_dir, "block_features.csv")
    token_csv = os.path.join(run_dir, "token_features.csv")
    summary_json = os.path.join(run_dir, "run_summary.json")

    write_csv(block_csv, block_rows)
    write_csv(token_csv, token_rows)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "summary": summary,
                "block_csv": block_csv,
                "token_csv": token_csv,
            },
            f,
            indent=2,
        )

    print(f"run_dir: {run_dir}")
    print(f"block_csv: {block_csv}")
    print(f"token_csv: {token_csv}")
    print(f"summary_json: {summary_json}")
    print(f"valid_blocks: {summary['valid_blocks']}")
    print(f"rejected_blocks: {summary['rejected_blocks']}")
    print(f"block_reject_rate: {summary['block_reject_rate']:.6f}")
    print(f"valid_tokens: {summary['valid_tokens']}")
    print(f"rejected_tokens: {summary['rejected_tokens']}")
    print(f"token_reject_rate: {summary['token_reject_rate']:.6f}")


if __name__ == "__main__":
    main()
