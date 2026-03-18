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

from MagicDec.Data.data_converter import convert_pg19_dataset
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
    parser.add_argument("--estimate_ratio", type=float, default=0.25)
    parser.add_argument("--num_max_token", type=int, default=64)
    parser.add_argument("--num_eval_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable_dynamic_budget", action="store_true")
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
    stage_path = os.path.join(log_dir, "stage_outputs.json")

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
        "total_tokens_generated",
    ]

    with open(step_path, "w", newline="") as f:
        csv.writer(f).writerow(step_headers)
    with open(acc_path, "w", newline="") as f:
        csv.writer(f).writerow(acc_headers)
    with open(stage_path, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

    return step_path, acc_path, stage_path


def append_stage_outputs(stage_path, entries):
    with open(stage_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.extend(entries)
    with open(stage_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_csv(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def empty_like_tokens(ref_token):
    return torch.empty((ref_token.shape[0], 0), dtype=ref_token.dtype, device=ref_token.device)


def load_longbench_config():
    base = os.path.join(PROJECT_ROOT, "Engine", "RetrievalAttention_new", "benchmark", "longbench", "config")
    with open(os.path.join(base, "model2path.json"), "r", encoding="utf-8") as f:
        model2path = json.load(f)
    with open(os.path.join(base, "model2maxlen.json"), "r", encoding="utf-8") as f:
        model2maxlen = json.load(f)
    with open(os.path.join(base, "dataset2prompt.json"), "r", encoding="utf-8") as f:
        dataset2prompt = json.load(f)
    return model2path, model2maxlen, dataset2prompt


def main():
    args = parse_args()
    setup_seed(args.seed)

    model2path, model2maxlen, dataset2prompt = load_longbench_config()

    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_device = "auto" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = model2path[args.model_name]
    max_length = model2maxlen[args.model_name]
    prompt_format = dataset2prompt[args.task]

    print(f"Using runtime_device={runtime_device}, model_device={model_device}")

    engine = LMBackend(dtype=torch.bfloat16, device=runtime_device, dec_len=args.gamma1 + 1)
    engine.load_model(model_path, max_length, torch.bfloat16, model_device, args.B)

    tokenizer = engine.model.tokenizer
    eos_id = tokenizer.eos_token_id

    if args.dataset == "pg19":
        dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
    else:
        dataset = load_dataset("THUDM/LongBench", args.task, split="test", trust_remote_code=True)

    log_dir = args.logs_dir
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(PROJECT_ROOT, log_dir)
    step_path, acc_path, stage_path = init_logs(log_dir)

    total_speculate_calls = 0
    total_early_verify_calls = 0
    total_final_verify_calls = 0
    total_tokens_generated = 0

    total_steps = min(args.num_eval_steps, len(dataset))

    for step in tqdm(range(total_steps), total=total_steps):
        batch = dataset[step]
        input_ids = engine.preprocess_input(
            batch,
            prompt_format,
            args.dataset,
        )

        attention_masks = engine.attention_masks

        step_speculate_calls = 0
        step_early_verify_calls = 0
        step_final_verify_calls = 0
        step_tokens_generated = 0
        stage_entries = []

        drafted_tokens_count = 0
        early_verified_accepted_count = 0
        early_bonus_count = 0
        final_authoritative_bonus_count = 0
        final_settled_tokens_count = 0
        unsettled_total = 0

        final_prerun_calls = 0

        engine.setup_caches(
            input_ids=input_ids,
            attention_masks=attention_masks,
            budget1=args.budget1,
            budget2=args.budget2,
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
        chunk_start_final_snapshot = engine.snapshot_state("final_verify")

        while len(emitted_tokens) < args.num_max_token:
            draft_snapshot = engine.snapshot_state("draft")
            early_snapshot = engine.snapshot_state("early_verify")

            t_draft = time.perf_counter()
            draft_tokens = engine.speculate(current_token, args.gamma1)
            elapsed_draft = time.perf_counter() - t_draft
            step_speculate_calls += 1
            drafted_tokens_count += draft_tokens.shape[1]
            stage_entries.append({"stage": "draft", "outputs": draft_tokens[0].detach().cpu().tolist()})

            t_early = time.perf_counter()
            early_outputs = engine.early_verify(current_token, draft_tokens)
            elapsed_early = time.perf_counter() - t_early
            step_early_verify_calls += 1
            stage_entries.append({"stage": "early verify", "outputs": early_outputs[0].detach().cpu().tolist()})

            accepted_len = 0
            for idx in range(args.gamma1):
                if int(draft_tokens[0, idx]) == int(early_outputs[0, idx]) and int(draft_tokens[0, idx]) != eos_id:
                    accepted_len += 1
                else:
                    break

            rejected_len = args.gamma1 - accepted_len
            early_bonus = early_outputs[:, accepted_len:accepted_len + 1]
            committed_online = torch.cat([draft_tokens[:, :accepted_len], early_bonus], dim=1)
            early_verified_accepted_count += accepted_len
            early_bonus_count += 1

            engine.revert_to("draft", draft_snapshot)
            engine.revert_to("early_verify", early_snapshot)

            accepted_prefix = committed_online[:, :-1]
            engine.commit_prefix("draft", current_token, accepted_prefix)
            engine.commit_prefix("early_verify", current_token, accepted_prefix)

            current_token = committed_online[:, -1:]
            pending_online_tokens = torch.cat([pending_online_tokens, committed_online], dim=1)

            print(
                f"[Stage Draft] elapsed={elapsed_draft:.4f}s accepted={accepted_len} rejected={rejected_len} "
                f"unsettled={unsettled_total} budget={args.budget1} text={to_text(tokenizer, draft_tokens)}"
            )
            print(
                f"[Stage Early Verify] elapsed={elapsed_early:.4f}s accepted={accepted_len} rejected={rejected_len} "
                f"unsettled={unsettled_total} budget={args.budget2} text={to_text(tokenizer, early_outputs)}"
            )

            should_settle = (
                pending_online_tokens.shape[1] >= args.gamma2
                or int(current_token[0, 0]) == eos_id
                or (step_tokens_generated + pending_online_tokens.shape[1]) >= args.num_max_token
            )

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
                stage_entries.append(
                    {
                        "stage": "final verify",
                        "outputs": final_outputs[0].detach().cpu().tolist(),
                    }
                )

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
                engine.revert_to("final_verify", chunk_start_final_snapshot)

                if authoritative_tokens.numel() == 0:
                    break

                final_prefix = authoritative_tokens[:, :-1]
                engine.commit_prefix("final_verify", chunk_start_final_token, final_prefix)
                engine.commit_prefix("early_verify", chunk_start_token, final_prefix, sync_source="final_verify")
                engine.commit_prefix("draft", chunk_start_token, final_prefix, sync_source="final_verify")
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
                chunk_start_final_snapshot = engine.snapshot_state("final_verify")

            if int(current_token[0, 0]) == eos_id:
                break

        total_speculate_calls += step_speculate_calls
        total_early_verify_calls += step_early_verify_calls
        total_final_verify_calls += step_final_verify_calls
        total_tokens_generated += step_tokens_generated

        print(f"=== Step {step} Statistics ===")
        print(f"Dynamic budget enabled: {args.enable_dynamic_budget}")
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
        print(f"Final verify cache state: {engine.cache_state_report('final_verify')}")

        print(f"=== Accumulated Statistics (up to step {step}) ===")
        print(f"Total speculate calls: {total_speculate_calls}")
        print(f"Total early verify calls: {total_early_verify_calls}")
        print(f"Total final verify calls: {total_final_verify_calls}")
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
                step_tokens_generated,
            ],
        )

        append_csv(
            acc_path,
            [
                step,
                args.dataset,
                args.prefix_len,
                args.gamma1,
                args.gamma2,
                args.budget1,
                args.budget2,
                total_speculate_calls,
                total_early_verify_calls,
                total_final_verify_calls,
                total_tokens_generated,
            ],
        )

        append_stage_outputs(stage_path, stage_entries)
        engine.clear_kv()

    append_csv(
        acc_path,
        [
            "final",
            args.dataset,
            args.prefix_len,
            args.gamma1,
            args.gamma2,
            args.budget1,
            args.budget2,
            total_speculate_calls,
            total_early_verify_calls,
            total_final_verify_calls,
            total_tokens_generated,
        ],
    )

    print("=== Final Accumulated Statistics ===")
    print(f"Total speculate calls: {total_speculate_calls}")
    print(f"Total early verify calls: {total_early_verify_calls}")
    print(f"Total final verify calls: {total_final_verify_calls}")
    print(f"Total tokens generated: {total_tokens_generated}")


if __name__ == "__main__":
    main()
