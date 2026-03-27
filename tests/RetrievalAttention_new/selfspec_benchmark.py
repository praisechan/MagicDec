import argparse
import csv
import json
import os
import sys
import time

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
    parser = argparse.ArgumentParser(description="RetrievalAttention_new 3-stage self-spec benchmark")
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3.1-8B")
    parser.add_argument("--dataset", type=str, default="pg19", choices=["pg19", "longbenchv1"])
    parser.add_argument("--task", type=str, default="gov_report")
    parser.add_argument("--attn_type", type=str, default="RetroInfer")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--prefix_len", type=int, default=8192)
    parser.add_argument("--gamma1", type=int, default=8)
    parser.add_argument("--gamma2", type=int, default=16)
    parser.add_argument("--budget1", type=float, default=0.05)
    parser.add_argument("--budget2", type=float, default=0.25)
    parser.add_argument("--budget2_high", type=float, default=0.5)
    parser.add_argument("--estimate_ratio", type=float, default=0.25)
    parser.add_argument("--num_max_token", type=int, default=64)
    parser.add_argument("--num_eval_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable_dynamic_budget", action="store_true")
    parser.add_argument("--T_low", type=float, default=0.05)
    parser.add_argument("--T_high", type=float, default=0.20)
    parser.add_argument("--logs_dir", type=str, default="MagicDec/tests/RetrievalAttention_new/logs/selfspec_3stage")
    return parser.parse_args()


def first_mismatch_idx(lhs, rhs):
    max_len = min(lhs.shape[1], rhs.shape[1])
    for idx in range(max_len):
        if int(lhs[0, idx]) != int(rhs[0, idx]):
            return idx
    return -1


def to_text(tokenizer, token_tensor):
    if token_tensor.numel() == 0:
        return ""
    return tokenizer.decode(token_tensor[0].detach().cpu().tolist(), skip_special_tokens=True)


def init_logs(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    step_path = os.path.join(log_dir, "step_log.csv")
    acc_path = os.path.join(log_dir, "accumulated_log.csv")

    step_headers = [
        "step",
        "dataset",
        "prefix_len",
        "gamma1",
        "gamma2",
        "budget1",
        "budget2",
        "speculate_calls",
        "early_verify_calls",
        "final_verify_calls",
        "skip_switches",
        "high_switches",
        "tokens_generated",
    ]
    acc_headers = [
        "step",
        "dataset",
        "prefix_len",
        "gamma1",
        "gamma2",
        "budget1",
        "budget2",
        "total_speculate_calls",
        "total_early_verify_calls",
        "total_final_verify_calls",
        "total_skip_switches",
        "total_high_switches",
        "total_tokens_generated",
        "model_name",
        "task",
        "budget2_high",
        "enable_dynamic_budget",
        "T_high",
        "T_low",
    ]

    if not os.path.exists(step_path):
        with open(step_path, "w", newline="") as f:
            csv.writer(f).writerow(step_headers)
    if not os.path.exists(acc_path):
        with open(acc_path, "w", newline="") as f:
            csv.writer(f).writerow(acc_headers)
    return step_path, acc_path


def append_csv(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def empty_like_tokens(ref_token):
    return torch.empty((ref_token.shape[0], 0), dtype=ref_token.dtype, device=ref_token.device)


def first_eos_idx(tokens, eos_id):
    for idx in range(tokens.shape[1]):
        if int(tokens[0, idx]) == eos_id:
            return idx
    return -1


def build_verified_commit(verify_tokens, verify_outputs, eos_id):
    accepted_len = 0
    verify_len = verify_tokens.shape[1]
    for idx in range(verify_len):
        if int(verify_tokens[0, idx]) == int(verify_outputs[0, idx]) and int(verify_tokens[0, idx]) != eos_id:
            accepted_len += 1
        else:
            break

    rejected_len = verify_len - accepted_len
    verify_bonus = verify_outputs[:, accepted_len:accepted_len + 1]
    committed_online = torch.cat([verify_tokens[:, :accepted_len], verify_bonus], dim=1)
    return accepted_len, rejected_len, committed_online


def replay_verified_prefix(engine, draft_snapshot, early_snapshot, early_high_snapshot, verify_start_token, committed_online, mode):
    engine.revert_to("draft", draft_snapshot)
    engine.revert_to("early_verify", early_snapshot)
    engine.revert_to("early_verify_high", early_high_snapshot)

    accepted_prefix = committed_online[:, :-1]
    # Keep one canonical replay path for cache growth. In high mode we still use
    # early_verify_high for acceptance decisions, but we replay committed prefix
    # through early_verify and synchronize to other caches to avoid stage drift.
    engine.commit_prefix("draft", verify_start_token, accepted_prefix, replay_source="early_verify")
    engine.commit_prefix(
        "early_verify_high",
        verify_start_token,
        accepted_prefix,
        sync_source="early_verify",
        source_replayed=True,
    )


def reset_skip_buffer(current_token):
    return None, None, empty_like_tokens(current_token), 0, 0


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
    else:
        prompt_format = dataset2prompt[args.task]

    print(f"Using runtime_device={runtime_device}, model_device={model_device}")

    engine = LMBackend(dtype=torch.bfloat16, device=runtime_device, dec_len=args.gamma1 + 1)
    engine.load_model(model_path, max_length, torch.bfloat16, model_device, args.B)

    tokenizer = engine.model.tokenizer
    eos_id = tokenizer.eos_token_id

    if args.dataset == "pg19":
        dataset = load_pg19_dataset()
    else:
        dataset = load_dataset("THUDM/LongBench", args.task, split="test", trust_remote_code=True)

    log_dir = args.logs_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(PROJECT_ROOT, log_dir)
    step_path, acc_path = init_logs(log_dir)

    total_speculate_calls = 0
    total_early_verify_calls = 0
    total_final_verify_calls = 0
    total_skip_switches = 0
    total_high_switches = 0
    total_tokens_generated = 0

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

        step_speculate_calls = 0
        step_early_verify_calls = 0
        step_final_verify_calls = 0
        step_tokens_generated = 0

        drafted_tokens_count = 0
        early_verified_accepted_count = 0
        early_bonus_count = 0
        final_authoritative_bonus_count = 0
        final_settled_tokens_count = 0
        unsettled_total = 0
        dynamic_skip_count = 0
        dynamic_high_count = 0
        dynamic_normal_count = 0
        dynamic_min_conf_values = []

        final_prerun_calls = 0

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

        while len(emitted_tokens) < args.num_max_token:
            # During a skip streak, always re-draft from the same committed anchor and
            # increase the drafted span length (gamma1, 2*gamma1, 3*gamma1, ...).
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
            draft_tokens, draft_confidence, min_conf = engine.speculate_with_confidence(draft_start_token, draft_decode_len)
            elapsed_draft = time.perf_counter() - t_draft
            drafted_tokens_count += draft_tokens.shape[1]

            dynamic_mode = "normal"
            if args.enable_dynamic_budget:
                dynamic_min_conf_values.append(min_conf)
                if min_conf > args.T_high:
                    dynamic_mode = "skip"
                    dynamic_skip_count += 1
                elif min_conf < args.T_low:
                    dynamic_mode = "high"
                    dynamic_high_count += 1
                else:
                    dynamic_mode = "normal"
                    dynamic_normal_count += 1
            else:
                dynamic_normal_count += 1

            # For skip streaks we repeatedly re-draft from the same anchor, so count only
            # the final stacked span length (replace prior counted length, do not accumulate).
            if dynamic_mode == "skip":
                step_speculate_calls += draft_decode_len - skip_accounted_draft_len
                skip_accounted_draft_len = draft_decode_len
            else:
                if consecutive_skip_count > 0:
                    step_speculate_calls += draft_decode_len - skip_accounted_draft_len
                else:
                    step_speculate_calls += draft_decode_len
                skip_accounted_draft_len = 0

            committed_online = None
            accepted_len = 0
            rejected_len = 0
            elapsed_early = 0.0

            if dynamic_mode == "skip":
                if consecutive_skip_count == 0:
                    # First skip in this buffered sequence: lock anchor state.
                    skip_anchor_token = current_token.clone()
                    skip_anchor_draft_snapshot = draft_snapshot

                eos_idx = first_eos_idx(draft_tokens, eos_id)
                accepted_len = (eos_idx + 1) if eos_idx >= 0 else draft_tokens.shape[1]
                rejected_len = draft_tokens.shape[1] - accepted_len
                skip_stacked_draft_tokens = draft_tokens[:, :accepted_len]
                consecutive_skip_count += 1

                if skip_stacked_draft_tokens.numel() == 0:
                    break
                # Skip mode must not commit to verifier caches; keep only draft rewound.
                engine.revert_to("draft", draft_snapshot)

                # Safety gate: if buffered skip span reaches settlement boundary,
                # run one normal early-verify pass before final settlement.
                skip_next_online_len = pending_online_tokens.shape[1] + skip_stacked_draft_tokens.shape[1]
                skip_next_total_generated = step_tokens_generated + skip_next_online_len
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
                    step_early_verify_calls += 1

                    accepted_len, rejected_len, committed_online = build_verified_commit(
                        verify_tokens,
                        early_outputs,
                        eos_id,
                    )
                    early_verified_accepted_count += accepted_len
                    early_bonus_count += 1

                    replay_verified_prefix(
                        engine,
                        skip_anchor_draft_snapshot,
                        early_snapshot,
                        early_high_snapshot,
                        verify_start_token,
                        committed_online,
                        "normal",
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
                # Normal/high after skips: verify the full stacked span in one pass.
                verify_start_token = skip_anchor_token if consecutive_skip_count > 0 else current_token
                verify_tokens = draft_tokens
                early_snapshot = engine.snapshot_state("early_verify")
                early_high_snapshot = engine.snapshot_state("early_verify_high")
                t_early = time.perf_counter()
                early_outputs = engine.early_verify(verify_start_token, verify_tokens, mode=dynamic_mode)
                elapsed_early = time.perf_counter() - t_early
                step_early_verify_calls += 1

                accepted_len, rejected_len, committed_online = build_verified_commit(
                    verify_tokens,
                    early_outputs,
                    eos_id,
                )
                early_verified_accepted_count += accepted_len
                early_bonus_count += 1

                # Commit only after batched verify decides accepted-prefix + bonus token.
                replay_verified_prefix(
                    engine,
                    skip_anchor_draft_snapshot if consecutive_skip_count > 0 else draft_snapshot,
                    early_snapshot,
                    early_high_snapshot,
                    verify_start_token,
                    committed_online,
                    dynamic_mode,
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

            # Evaluate settlement is needed
            next_online_len = pending_online_tokens.shape[1] + proposed_online_tokens.shape[1]
            next_total_generated = step_tokens_generated + next_online_len
            proposed_last_token = int(proposed_online_tokens[0, -1])
            should_settle = (
                next_online_len >= args.gamma2
                or proposed_last_token == eos_id
                or next_total_generated >= args.num_max_token
            )

            # In normal/high mode we materialize immediately.
            # In skip mode we materialize only at settlement boundary.
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
                remaining_budget = args.num_max_token - step_tokens_generated
                if remaining_budget <= 0:
                    break

                t_final = time.perf_counter()
                online_span = pending_online_tokens[:, :remaining_budget]
                if online_span.numel() == 0:
                    break

                final_outputs = engine.final_verify(final_current_token, online_span)
                step_final_verify_calls += 1

                mismatch = first_mismatch_idx(online_span, final_outputs[:, :online_span.shape[1]])
                accepted_len = online_span.shape[1] if mismatch < 0 else mismatch
                if mismatch >= 0:
                    unsettled_total += 1
                final_authoritative_bonus_count += 1
                authoritative_tokens = torch.cat(
                    [online_span[:, :accepted_len], final_outputs[:, accepted_len:accepted_len + 1]],
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
                step_tokens_generated += len(committed_list)
                final_settled_tokens_count += len(committed_list)

                final_rejected = online_span.shape[1] - accepted_len
                elapsed_final = time.perf_counter() - t_final
                print(
                    f"[Stage Final Verify] elapsed={elapsed_final:.4f}s accepted={authoritative_tokens.shape[1]} "
                    f"rejected={final_rejected} unsettled={unsettled_total} budget=1.0 "
                    f"text={to_text(tokenizer, final_outputs)}"
                )

                pending_online_tokens = empty_like_tokens(current_token)
                chunk_start_token = current_token.clone()
                chunk_start_final_token = final_current_token.clone()
                chunk_start_draft_snapshot = engine.snapshot_state("draft")
                chunk_start_early_snapshot = engine.snapshot_state("early_verify")
                chunk_start_early_high_snapshot = engine.snapshot_state("early_verify_high")
                chunk_start_final_snapshot = engine.snapshot_state("final_verify")

            if int(current_token[0, 0]) == eos_id:
                break

        total_speculate_calls += step_speculate_calls
        total_early_verify_calls += step_early_verify_calls
        total_final_verify_calls += step_final_verify_calls
        total_skip_switches += dynamic_skip_count
        total_high_switches += dynamic_high_count
        total_tokens_generated += step_tokens_generated

        print(f"=== Step {step} Statistics ===")
        print(f"Dynamic budget enabled: {args.enable_dynamic_budget}")
        if args.enable_dynamic_budget:
            min_conf_stat = min(dynamic_min_conf_values) if dynamic_min_conf_values else float("inf")
            print(f"Dynamic route counters: skip={dynamic_skip_count}, high={dynamic_high_count}, normal={dynamic_normal_count}")
            print(f"Minimum confidence observed: {min_conf_stat:.6f}")
        print(f"Speculate calls: {step_speculate_calls}")
        print(f"Early Verify calls: {step_early_verify_calls}")
        print(f"Final Verify calls: {step_final_verify_calls}")
        print(f"Final Verify pre-run calls: {final_prerun_calls}")
        print(f"Tokens generated: {step_tokens_generated}")
        print(f"Drafted tokens: {drafted_tokens_count}")
        print(f"Early verified accepted: {early_verified_accepted_count}")
        print(f"Final settled tokens: {final_settled_tokens_count}")
        print(f"Early bonus tokens: {early_bonus_count}")
        print(f"Final authoritative bonus tokens: {final_authoritative_bonus_count}")
        print(f"Emitted tokens: {step_tokens_generated}")
        print(f"Draft cache state: {engine.cache_state_report('draft')}")
        print(f"Early verify cache state: {engine.cache_state_report('early_verify')}")
        print(f"Early verify high cache state: {engine.cache_state_report('early_verify_high')}")
        print(f"Final verify cache state: {engine.cache_state_report('final_verify')}")

        print(f"=== Accumulated Statistics (up to step {step}) ===")
        print(f"Total speculate calls: {total_speculate_calls}")
        print(f"Total early verify calls: {total_early_verify_calls}")
        print(f"Total final verify calls: {total_final_verify_calls}")
        print(f"Total skip switches: {total_skip_switches}")
        print(f"Total high switches: {total_high_switches}")
        print(f"Total tokens generated: {total_tokens_generated}")

        append_csv(
            step_path,
            [
                step,
                args.dataset,
                args.prefix_len,
                args.gamma1,
                args.gamma2,
                args.budget1,
                args.budget2,
                step_speculate_calls,
                step_early_verify_calls,
                step_final_verify_calls,
                dynamic_skip_count,
                dynamic_high_count,
                step_tokens_generated,
                args.model_name,
                args.task,
                args.budget2_high,
                args.enable_dynamic_budget,
                args.T_high,
                args.T_low,
            ],
        )

        engine.clear_kv()

    append_csv(
        acc_path,
        [
            args.num_eval_steps,
            args.dataset,
            args.prefix_len,
            args.gamma1,
            args.gamma2,
            args.budget1,
            args.budget2,
            total_speculate_calls,
            total_early_verify_calls,
            total_final_verify_calls,
            total_skip_switches,
            total_high_switches,
            total_tokens_generated,
            args.model_name,
            args.task,
            args.budget2_high,
            args.enable_dynamic_budget,
            args.T_high,
            args.T_low,
        ],
    )

    print("=== Final Accumulated Statistics ===")
    print(f"Total speculate calls: {total_speculate_calls}")
    print(f"Total early verify calls: {total_early_verify_calls}")
    print(f"Total final verify calls: {total_final_verify_calls}")
    print(f"Total skip switches: {total_skip_switches}")
    print(f"Total high switches: {total_high_switches}")
    print(f"Total tokens generated: {total_tokens_generated}")


if __name__ == "__main__":
    main()
