import argparse
import csv
import json
import os
import sys
import time
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
    RECOMMENDED_DRAFT_RULE,
    build_verified_commit,
    compute_cost_breakdown,
    compute_decode_weighted_cost_breakdown,
    compute_risk_score,
    empty_like_tokens,
    first_eos_idx,
    first_mismatch_idx,
    format_float_tag,
    get_pg19_prompt_format,
    load_longbench_config,
    load_pg19_dataset,
    replay_verified_prefix,
    reset_skip_buffer,
    sanitize_tag,
    summarize_draft_features,
)


RISK_RULE_CHOICES = [
    "baseline_min_gap",
    "mean_margin",
    "last_margin",
    "min_margin_mean_margin",
    "min_margin_low_count",
    "min_margin_early_position",
    "min_margin_mean_early",
]


def parse_args():
    parser = argparse.ArgumentParser(description="RetrievalAttention_new 3-stage self-spec benchmark with risk-gated high routing")
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3.1-8B")
    parser.add_argument("--dataset", type=str, default="pg19", choices=["pg19", "longbenchv1"])
    parser.add_argument("--task", type=str, default="gov_report")
    parser.add_argument("--attn_type", type=str, default="RetroInfer")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--prefix_len", type=int, default=8192)
    parser.add_argument("--gamma1", type=int, default=6)
    parser.add_argument("--gamma2", type=int, default=32)
    parser.add_argument("--budget1", type=float, default=0.02)
    parser.add_argument("--budget2", type=float, default=0.10)
    parser.add_argument("--budget2_high", type=float, default=0.25)
    parser.add_argument("--estimate_ratio", type=float, default=0.25)
    parser.add_argument("--num_max_token", type=int, default=100)
    parser.add_argument("--num_eval_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable_dynamic_budget", action="store_true")
    parser.add_argument("--T_low", type=float, default=0.05)
    parser.add_argument("--T_high", type=float, default=0.20)
    parser.add_argument("--risk_rule", type=str, default=RECOMMENDED_DRAFT_RULE, choices=RISK_RULE_CHOICES)
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="tests/RetrievalAttention_new/logs/selfspec_riskgate",
    )
    return parser.parse_args()


def to_text(tokenizer, token_tensor):
    if token_tensor.numel() == 0:
        return ""
    return tokenizer.decode(token_tensor[0].detach().cpu().tolist(), skip_special_tokens=True)


def truncate_feature_dict(feature_dict, length):
    return {name: tensor[:, :length] for name, tensor in feature_dict.items()}


def build_run_dir(base_dir, args):
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(PROJECT_ROOT, base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"riskgate_{sanitize_tag(args.dataset)}"
        f"_task_{sanitize_tag(args.task)}"
        f"_rule_{sanitize_tag(args.risk_rule)}"
        f"_steps_{sanitize_tag(args.num_eval_steps)}"
        f"_tokens_{sanitize_tag(args.num_max_token)}"
        f"_g1_{sanitize_tag(args.gamma1)}"
        f"_g2_{sanitize_tag(args.gamma2)}"
        f"_b1_{format_float_tag(args.budget1)}"
        f"_b2_{format_float_tag(args.budget2)}"
        f"_b2h_{format_float_tag(args.budget2_high)}"
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


def select_dynamic_mode(args, block_features):
    min_margin = float(block_features["block_min_margin"])
    if not args.enable_dynamic_budget:
        return "normal", min_margin, float("nan")

    if min_margin > args.T_high:
        return "skip", min_margin, float("nan")

    if args.risk_rule == "baseline_min_gap":
        risk_score = 1.0 - min_margin
        dynamic_mode = "high" if min_margin < args.T_low else "normal"
    else:
        risk_score = compute_risk_score(block_features, args.risk_rule)
        dynamic_mode = "high" if risk_score >= args.T_low else "normal"
    return dynamic_mode, min_margin, risk_score


def main():
    args = parse_args()

    if args.risk_rule == "baseline_min_gap" and args.T_low > args.T_high:
        raise ValueError("For baseline_min_gap, T_low should not be greater than T_high.")

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

    print(f"Using runtime_device={runtime_device}, model_device={model_device}")
    print(f"Risk rule: {args.risk_rule}")

    engine = LMBackend(dtype=torch.bfloat16, device=runtime_device, dec_len=args.gamma1 + 1)
    engine.load_model(model_path, max_length, torch.bfloat16, model_device, args.B)

    tokenizer = engine.model.tokenizer
    eos_id = tokenizer.eos_token_id

    run_dir = build_run_dir(args.logs_dir, args)
    step_rows = []

    totals = {
        "draft_decode_calls": 0,
        "early_verify_invocations": 0,
        "early_normal_invocations": 0,
        "early_high_invocations": 0,
        "early_normal_decode_calls": 0,
        "early_high_decode_calls": 0,
        "final_verify_invocations": 0,
        "final_decode_calls": 0,
        "skip_switches": 0,
        "high_switches": 0,
        "tokens_generated": 0,
    }

    total_steps = min(args.num_eval_steps, len(dataset))

    for step in tqdm(range(total_steps), total=total_steps):
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
            max_new_tokens=args.num_max_token + args.gamma2 + args.gamma1 + 8,
        )
        engine.setup_final_verify_cache(
            input_ids=input_ids,
            attention_masks=attention_masks,
            max_new_tokens=args.num_max_token + args.gamma2 + args.gamma1 + 8,
        )

        current_token = engine.encode(input_ids=input_ids)
        emitted_tokens = []
        pending_online_tokens = empty_like_tokens(current_token)
        chunk_start_token = current_token.clone()
        final_current_token = engine.prefill_tokens["final_verify"].clone()
        chunk_start_final_token = final_current_token.clone()
        chunk_start_draft_snapshot = engine.snapshot_state("draft")
        chunk_start_early_snapshot = engine.snapshot_state("early_verify")
        chunk_start_early_high_snapshot = engine.snapshot_state("early_verify_high")
        chunk_start_final_snapshot = engine.snapshot_state("final_verify")

        skip_anchor_token = None
        skip_anchor_draft_snapshot = None
        skip_stacked_draft_tokens = empty_like_tokens(current_token)
        consecutive_skip_count = 0
        skip_accounted_draft_len = 0

        step_stats = {
            "draft_decode_calls": 0,
            "early_verify_invocations": 0,
            "early_normal_invocations": 0,
            "early_high_invocations": 0,
            "early_normal_decode_calls": 0,
            "early_high_decode_calls": 0,
            "final_verify_invocations": 0,
            "final_decode_calls": 0,
            "skip_switches": 0,
            "high_switches": 0,
            "tokens_generated": 0,
            "drafted_tokens": 0,
            "early_verified_accepted": 0,
            "early_bonus_tokens": 0,
            "final_authoritative_bonus_tokens": 0,
            "final_settled_tokens": 0,
            "unsettled_total": 0,
            "dynamic_normal_count": 0,
            "min_margin_values": [],
            "risk_score_values": [],
        }

        while len(emitted_tokens) < args.num_max_token:
            if consecutive_skip_count > 0:
                draft_decode_len = args.gamma1 * (consecutive_skip_count + 1)
                draft_start_token = skip_anchor_token
                engine.revert_to("draft", skip_anchor_draft_snapshot)
                draft_snapshot = skip_anchor_draft_snapshot
            else:
                draft_decode_len = args.gamma1
                draft_start_token = current_token
                draft_snapshot = engine.snapshot_state("draft")

            t_draft = time.perf_counter()
            draft_tokens, draft_feature_dict = engine.speculate_with_features(draft_start_token, draft_decode_len)
            elapsed_draft = time.perf_counter() - t_draft
            if draft_tokens.numel() == 0:
                break

            step_stats["drafted_tokens"] += draft_tokens.shape[1]
            block_features = summarize_draft_features(draft_feature_dict)
            dynamic_mode, min_margin, risk_score = select_dynamic_mode(args, block_features)
            step_stats["min_margin_values"].append(min_margin)
            if not torch.isnan(torch.tensor(risk_score)):
                step_stats["risk_score_values"].append(risk_score)

            if args.enable_dynamic_budget:
                if dynamic_mode == "skip":
                    step_stats["skip_switches"] += 1
                elif dynamic_mode == "high":
                    step_stats["high_switches"] += 1
                else:
                    step_stats["dynamic_normal_count"] += 1
            else:
                step_stats["dynamic_normal_count"] += 1

            if dynamic_mode == "skip":
                step_stats["draft_decode_calls"] += draft_decode_len - skip_accounted_draft_len
                skip_accounted_draft_len = draft_decode_len
            else:
                if consecutive_skip_count > 0:
                    step_stats["draft_decode_calls"] += draft_decode_len - skip_accounted_draft_len
                else:
                    step_stats["draft_decode_calls"] += draft_decode_len
                skip_accounted_draft_len = 0

            committed_online = None
            accepted_len = 0
            rejected_len = 0
            elapsed_early = 0.0

            if dynamic_mode == "skip":
                if consecutive_skip_count == 0:
                    skip_anchor_token = current_token.clone()
                    skip_anchor_draft_snapshot = draft_snapshot

                eos_idx = first_eos_idx(draft_tokens, eos_id)
                accepted_len = (eos_idx + 1) if eos_idx >= 0 else draft_tokens.shape[1]
                rejected_len = draft_tokens.shape[1] - accepted_len
                skip_stacked_draft_tokens = draft_tokens[:, :accepted_len]
                if eos_idx >= 0:
                    draft_feature_dict = truncate_feature_dict(draft_feature_dict, accepted_len)
                consecutive_skip_count += 1

                if skip_stacked_draft_tokens.numel() == 0:
                    break

                engine.revert_to("draft", draft_snapshot)

                skip_next_online_len = pending_online_tokens.shape[1] + skip_stacked_draft_tokens.shape[1]
                skip_next_total_generated = step_stats["tokens_generated"] + skip_next_online_len
                skip_last_token = int(skip_stacked_draft_tokens[0, -1])
                should_force_skip_safety_verify = (
                    skip_next_online_len >= args.gamma2
                    or skip_last_token == eos_id
                    or skip_next_total_generated >= args.num_max_token
                )

                if should_force_skip_safety_verify:
                    verify_start_token = skip_anchor_token
                    verify_tokens = skip_stacked_draft_tokens
                    early_snapshot = engine.snapshot_state("early_verify")
                    early_high_snapshot = engine.snapshot_state("early_verify_high")
                    t_early = time.perf_counter()
                    early_outputs = engine.early_verify(verify_start_token, verify_tokens, mode="normal")
                    elapsed_early = time.perf_counter() - t_early

                    step_stats["early_verify_invocations"] += 1
                    step_stats["early_normal_invocations"] += 1
                    step_stats["early_normal_decode_calls"] += verify_tokens.shape[1] + 1
                    accepted_len, rejected_len, committed_online = build_verified_commit(
                        verify_tokens,
                        early_outputs,
                        eos_id,
                    )
                    step_stats["early_verified_accepted"] += accepted_len
                    step_stats["early_bonus_tokens"] += 1

                    replay_verified_prefix(
                        engine,
                        skip_anchor_draft_snapshot,
                        early_snapshot,
                        early_high_snapshot,
                        verify_start_token,
                        committed_online,
                    )

                    (
                        skip_anchor_token,
                        skip_anchor_draft_snapshot,
                        skip_stacked_draft_tokens,
                        consecutive_skip_count,
                        skip_accounted_draft_len,
                    ) = reset_skip_buffer(current_token)

                    dynamic_mode = "normal"
            else:
                verify_start_token = skip_anchor_token if consecutive_skip_count > 0 else current_token
                verify_tokens = draft_tokens
                early_snapshot = engine.snapshot_state("early_verify")
                early_high_snapshot = engine.snapshot_state("early_verify_high")
                t_early = time.perf_counter()
                early_outputs = engine.early_verify(verify_start_token, verify_tokens, mode=dynamic_mode)
                elapsed_early = time.perf_counter() - t_early

                step_stats["early_verify_invocations"] += 1
                if dynamic_mode == "high":
                    step_stats["early_high_invocations"] += 1
                    step_stats["early_high_decode_calls"] += verify_tokens.shape[1] + 1
                else:
                    step_stats["early_normal_invocations"] += 1
                    step_stats["early_normal_decode_calls"] += verify_tokens.shape[1] + 1

                accepted_len, rejected_len, committed_online = build_verified_commit(
                    verify_tokens,
                    early_outputs,
                    eos_id,
                )
                step_stats["early_verified_accepted"] += accepted_len
                step_stats["early_bonus_tokens"] += 1

                replay_verified_prefix(
                    engine,
                    skip_anchor_draft_snapshot if consecutive_skip_count > 0 else draft_snapshot,
                    early_snapshot,
                    early_high_snapshot,
                    verify_start_token,
                    committed_online,
                )
                (
                    skip_anchor_token,
                    skip_anchor_draft_snapshot,
                    skip_stacked_draft_tokens,
                    consecutive_skip_count,
                    skip_accounted_draft_len,
                ) = reset_skip_buffer(current_token)

            proposed_online_tokens = skip_stacked_draft_tokens if dynamic_mode == "skip" else committed_online
            if proposed_online_tokens is None or proposed_online_tokens.numel() == 0:
                break

            next_online_len = pending_online_tokens.shape[1] + proposed_online_tokens.shape[1]
            next_total_generated = step_stats["tokens_generated"] + next_online_len
            proposed_last_token = int(proposed_online_tokens[0, -1])
            should_settle = (
                next_online_len >= args.gamma2
                or proposed_last_token == eos_id
                or next_total_generated >= args.num_max_token
            )

            should_materialize_pending = (dynamic_mode != "skip") or should_settle
            if should_materialize_pending:
                pending_online_tokens = torch.cat([pending_online_tokens, proposed_online_tokens], dim=1)
                current_token = proposed_online_tokens[:, -1:]
                if dynamic_mode == "skip":
                    (
                        skip_anchor_token,
                        skip_anchor_draft_snapshot,
                        skip_stacked_draft_tokens,
                        consecutive_skip_count,
                        skip_accounted_draft_len,
                    ) = reset_skip_buffer(current_token)

            if should_settle:
                remaining_budget = args.num_max_token - step_stats["tokens_generated"]
                if remaining_budget <= 0:
                    break

                t_final = time.perf_counter()
                online_span = pending_online_tokens[:, :remaining_budget]
                if online_span.numel() == 0:
                    break

                final_outputs = engine.final_verify(final_current_token, online_span)
                step_stats["final_verify_invocations"] += 1
                step_stats["final_decode_calls"] += online_span.shape[1] + 1

                mismatch = first_mismatch_idx(online_span, final_outputs[:, : online_span.shape[1]])
                accepted_len = online_span.shape[1] if mismatch < 0 else mismatch
                if mismatch >= 0:
                    step_stats["unsettled_total"] += 1
                step_stats["final_authoritative_bonus_tokens"] += 1

                authoritative_tokens = torch.cat(
                    [online_span[:, :accepted_len], final_outputs[:, accepted_len : accepted_len + 1]],
                    dim=1,
                )
                authoritative_tokens = authoritative_tokens[:, :remaining_budget]

                engine.revert_to("draft", chunk_start_draft_snapshot)
                engine.revert_to("early_verify", chunk_start_early_snapshot)
                engine.revert_to("early_verify_high", chunk_start_early_high_snapshot)
                engine.revert_to("final_verify", chunk_start_final_snapshot)

                if authoritative_tokens.numel() == 0:
                    break

                final_prefix = authoritative_tokens[:, :-1]
                engine.commit_prefix("final_verify", chunk_start_final_token, final_prefix)
                engine.commit_prefix(
                    "early_verify",
                    chunk_start_token,
                    final_prefix,
                    sync_source="final_verify",
                    source_replayed=True,
                )
                engine.commit_prefix(
                    "early_verify_high",
                    chunk_start_token,
                    final_prefix,
                    sync_source="final_verify",
                    source_replayed=True,
                )
                engine.commit_prefix(
                    "draft",
                    chunk_start_token,
                    final_prefix,
                    sync_source="final_verify",
                    source_replayed=True,
                )

                current_token = authoritative_tokens[:, -1:]
                final_current_token = authoritative_tokens[:, -1:]

                committed_list = authoritative_tokens[0].detach().cpu().tolist()
                emitted_tokens.extend(committed_list)
                step_stats["tokens_generated"] += len(committed_list)
                step_stats["final_settled_tokens"] += len(committed_list)

                final_rejected = online_span.shape[1] - accepted_len
                elapsed_final = time.perf_counter() - t_final
                print(
                    f"[Stage Final Verify] elapsed={elapsed_final:.4f}s accepted={authoritative_tokens.shape[1]} "
                    f"rejected={final_rejected} unsettled={step_stats['unsettled_total']} budget=1.0 "
                    f"text={to_text(tokenizer, final_outputs)}"
                )

                pending_online_tokens = empty_like_tokens(current_token)
                chunk_start_token = current_token.clone()
                chunk_start_final_token = final_current_token.clone()
                chunk_start_draft_snapshot = engine.snapshot_state("draft")
                chunk_start_early_snapshot = engine.snapshot_state("early_verify")
                chunk_start_early_high_snapshot = engine.snapshot_state("early_verify_high")
                chunk_start_final_snapshot = engine.snapshot_state("final_verify")

            print(
                f"[Stage Draft/Early] mode={dynamic_mode} elapsed_draft={elapsed_draft:.4f}s "
                f"elapsed_early={elapsed_early:.4f}s accepted={accepted_len} rejected={rejected_len} "
                f"min_margin={min_margin:.6f} "
                f"risk_score={'nan' if torch.isnan(torch.tensor(risk_score)) else f'{risk_score:.6f}'}"
            )

            if int(current_token[0, 0]) == eos_id:
                break

        totals["draft_decode_calls"] += step_stats["draft_decode_calls"]
        totals["early_verify_invocations"] += step_stats["early_verify_invocations"]
        totals["early_normal_invocations"] += step_stats["early_normal_invocations"]
        totals["early_high_invocations"] += step_stats["early_high_invocations"]
        totals["early_normal_decode_calls"] += step_stats["early_normal_decode_calls"]
        totals["early_high_decode_calls"] += step_stats["early_high_decode_calls"]
        totals["final_verify_invocations"] += step_stats["final_verify_invocations"]
        totals["final_decode_calls"] += step_stats["final_decode_calls"]
        totals["skip_switches"] += step_stats["skip_switches"]
        totals["high_switches"] += step_stats["high_switches"]
        totals["tokens_generated"] += step_stats["tokens_generated"]

        step_cost = compute_cost_breakdown(
            args.budget1,
            args.budget2,
            args.budget2_high,
            step_stats["draft_decode_calls"],
            step_stats["early_normal_invocations"],
            step_stats["early_high_invocations"],
            step_stats["final_verify_invocations"],
        )
        step_decode_weighted_cost = compute_decode_weighted_cost_breakdown(
            args.budget1,
            args.budget2,
            args.budget2_high,
            step_stats["draft_decode_calls"],
            step_stats["early_normal_decode_calls"],
            step_stats["early_high_decode_calls"],
            step_stats["final_decode_calls"],
        )

        min_margin_stat = min(step_stats["min_margin_values"]) if step_stats["min_margin_values"] else float("inf")
        risk_score_stat = max(step_stats["risk_score_values"]) if step_stats["risk_score_values"] else float("nan")

        print(f"=== Step {step} Statistics ===")
        print(f"Dynamic budget enabled: {args.enable_dynamic_budget}")
        print(f"Risk rule: {args.risk_rule}")
        print(
            f"Dynamic route counters: skip={step_stats['skip_switches']}, "
            f"high={step_stats['high_switches']}, normal={step_stats['dynamic_normal_count']}"
        )
        print(f"Minimum margin observed: {min_margin_stat:.6f}")
        if step_stats["risk_score_values"]:
            print(f"Maximum risk score observed: {risk_score_stat:.6f}")
        print(f"Draft decode calls: {step_stats['draft_decode_calls']}")
        print(f"Early verify invocations: {step_stats['early_verify_invocations']}")
        print(f"Early normal invocations: {step_stats['early_normal_invocations']}")
        print(f"Early high invocations: {step_stats['early_high_invocations']}")
        print(f"Early normal decode calls: {step_stats['early_normal_decode_calls']}")
        print(f"Early high decode calls: {step_stats['early_high_decode_calls']}")
        print(f"Final verify invocations: {step_stats['final_verify_invocations']}")
        print(f"Final decode calls: {step_stats['final_decode_calls']}")
        print(f"Tokens generated: {step_stats['tokens_generated']}")
        print(f"Drafted tokens: {step_stats['drafted_tokens']}")
        print(f"Early verified accepted: {step_stats['early_verified_accepted']}")
        print(f"Final settled tokens: {step_stats['final_settled_tokens']}")
        print(f"Early bonus tokens: {step_stats['early_bonus_tokens']}")
        print(f"Final authoritative bonus tokens: {step_stats['final_authoritative_bonus_tokens']}")
        print(
            f"Cost breakdown: draft={step_cost['draft_cost']:.4f}, "
            f"early_normal={step_cost['early_normal_cost']:.4f}, "
            f"early_high={step_cost['early_high_cost']:.4f}, "
            f"final={step_cost['final_cost']:.4f}, total={step_cost['total_cost']:.4f}"
        )
        print(
            f"Decode-weighted cost: draft={step_decode_weighted_cost['decode_weighted_draft_cost']:.4f}, "
            f"early_normal={step_decode_weighted_cost['decode_weighted_early_normal_cost']:.4f}, "
            f"early_high={step_decode_weighted_cost['decode_weighted_early_high_cost']:.4f}, "
            f"final={step_decode_weighted_cost['decode_weighted_final_cost']:.4f}, "
            f"total={step_decode_weighted_cost['decode_weighted_total_cost']:.4f}"
        )
        print(f"Draft cache state: {engine.cache_state_report('draft')}")
        print(f"Early verify cache state: {engine.cache_state_report('early_verify')}")
        print(f"Early verify high cache state: {engine.cache_state_report('early_verify_high')}")
        print(f"Final verify cache state: {engine.cache_state_report('final_verify')}")

        step_rows.append(
            {
                "step": step,
                "dataset": args.dataset,
                "task": args.task,
                "model_name": args.model_name,
                "prefix_len": args.prefix_len,
                "gamma1": args.gamma1,
                "gamma2": args.gamma2,
                "budget1": args.budget1,
                "budget2": args.budget2,
                "budget2_high": args.budget2_high,
                "enable_dynamic_budget": args.enable_dynamic_budget,
                "risk_rule": args.risk_rule,
                "T_low": args.T_low,
                "T_high": args.T_high,
                "draft_decode_calls": step_stats["draft_decode_calls"],
                "early_verify_invocations": step_stats["early_verify_invocations"],
                "early_normal_invocations": step_stats["early_normal_invocations"],
                "early_high_invocations": step_stats["early_high_invocations"],
                "early_normal_decode_calls": step_stats["early_normal_decode_calls"],
                "early_high_decode_calls": step_stats["early_high_decode_calls"],
                "final_verify_invocations": step_stats["final_verify_invocations"],
                "final_decode_calls": step_stats["final_decode_calls"],
                "skip_switches": step_stats["skip_switches"],
                "high_switches": step_stats["high_switches"],
                "tokens_generated": step_stats["tokens_generated"],
                **step_cost,
                **step_decode_weighted_cost,
            }
        )

        engine.clear_kv()

    total_cost = compute_cost_breakdown(
        args.budget1,
        args.budget2,
        args.budget2_high,
        totals["draft_decode_calls"],
        totals["early_normal_invocations"],
        totals["early_high_invocations"],
        totals["final_verify_invocations"],
    )
    total_decode_weighted_cost = compute_decode_weighted_cost_breakdown(
        args.budget1,
        args.budget2,
        args.budget2_high,
        totals["draft_decode_calls"],
        totals["early_normal_decode_calls"],
        totals["early_high_decode_calls"],
        totals["final_decode_calls"],
    )

    accumulated_row = {
        "steps": total_steps,
        "dataset": args.dataset,
        "task": args.task,
        "model_name": args.model_name,
        "prefix_len": args.prefix_len,
        "gamma1": args.gamma1,
        "gamma2": args.gamma2,
        "budget1": args.budget1,
        "budget2": args.budget2,
        "budget2_high": args.budget2_high,
        "enable_dynamic_budget": args.enable_dynamic_budget,
        "risk_rule": args.risk_rule,
        "T_low": args.T_low,
        "T_high": args.T_high,
        **totals,
        **total_cost,
        **total_decode_weighted_cost,
    }

    step_csv = os.path.join(run_dir, "step_log.csv")
    accumulated_csv = os.path.join(run_dir, "accumulated_log.csv")
    summary_json = os.path.join(run_dir, "run_summary.json")
    write_csv(step_csv, step_rows)
    write_csv(accumulated_csv, [accumulated_row])
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "cost_metric": {
                    "primary": "budget1*draft_decode_calls + budget2*early_normal_invocations + budget2_high*early_high_invocations + 1.0*final_verify_invocations",
                    "secondary": "decode-weighted stage costs are logged separately for debugging only",
                },
                "accumulated": accumulated_row,
            },
            f,
            indent=2,
        )

    print("=== Final Accumulated Statistics ===")
    print(f"Risk rule: {args.risk_rule}")
    print(f"Total draft decode calls: {totals['draft_decode_calls']}")
    print(f"Total early verify invocations: {totals['early_verify_invocations']}")
    print(f"Total early normal invocations: {totals['early_normal_invocations']}")
    print(f"Total early high invocations: {totals['early_high_invocations']}")
    print(f"Total early normal decode calls: {totals['early_normal_decode_calls']}")
    print(f"Total early high decode calls: {totals['early_high_decode_calls']}")
    print(f"Total final verify invocations: {totals['final_verify_invocations']}")
    print(f"Total final decode calls: {totals['final_decode_calls']}")
    print(f"Total skip switches: {totals['skip_switches']}")
    print(f"Total high switches: {totals['high_switches']}")
    print(f"Total tokens generated: {totals['tokens_generated']}")
    print(
        f"Final cost breakdown: draft={total_cost['draft_cost']:.4f}, "
        f"early_normal={total_cost['early_normal_cost']:.4f}, "
        f"early_high={total_cost['early_high_cost']:.4f}, "
        f"final={total_cost['final_cost']:.4f}, total={total_cost['total_cost']:.4f}"
    )
    print(
        f"Final decode-weighted cost: draft={total_decode_weighted_cost['decode_weighted_draft_cost']:.4f}, "
        f"early_normal={total_decode_weighted_cost['decode_weighted_early_normal_cost']:.4f}, "
        f"early_high={total_decode_weighted_cost['decode_weighted_early_high_cost']:.4f}, "
        f"final={total_decode_weighted_cost['decode_weighted_final_cost']:.4f}, "
        f"total={total_decode_weighted_cost['decode_weighted_total_cost']:.4f}"
    )
    print(f"run_dir: {run_dir}")
    print(f"step_csv: {step_csv}")
    print(f"accumulated_csv: {accumulated_csv}")
    print(f"summary_json: {summary_json}")


if __name__ == "__main__":
    main()
