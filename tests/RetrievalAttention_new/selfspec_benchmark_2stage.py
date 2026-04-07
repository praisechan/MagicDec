import argparse
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
from selfspec_benchmark import (
    append_csv,
    empty_like_tokens,
    first_eos_idx,
    first_mismatch_idx,
    get_pg19_prompt_format,
    init_logs,
    load_longbench_config,
    load_pg19_dataset,
    to_text,
)


def parse_args():
    parser = argparse.ArgumentParser(description="RetrievalAttention_new 2-stage self-spec benchmark")
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
    parser.add_argument("--logs_dir", type=str, default="MagicDec/tests/RetrievalAttention_new/logs/selfspec_2stage")
    return parser.parse_args()


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
    if args.enable_dynamic_budget:
        print("Dynamic budget routing is ignored in the 2-stage benchmark; skip/high counters will remain zero.")

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
        final_authoritative_bonus_count = 0
        final_settled_tokens_count = 0
        unsettled_total = 0
        dynamic_skip_count = 0
        dynamic_high_count = 0
        dynamic_normal_count = 0
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
        chunk_start_final_snapshot = engine.snapshot_state("final_verify")

        while len(emitted_tokens) < args.num_max_token:
            t_draft = time.perf_counter()
            draft_tokens = engine.speculate(current_token, args.gamma1)
            elapsed_draft = time.perf_counter() - t_draft

            eos_idx = first_eos_idx(draft_tokens, eos_id)
            proposed_online_tokens = draft_tokens if eos_idx < 0 else draft_tokens[:, :eos_idx + 1]
            if proposed_online_tokens.numel() == 0:
                break

            drafted_tokens_count += proposed_online_tokens.shape[1]
            step_speculate_calls += proposed_online_tokens.shape[1]
            dynamic_normal_count += 1

            next_online_len = pending_online_tokens.shape[1] + proposed_online_tokens.shape[1]
            next_total_generated = step_tokens_generated + next_online_len
            proposed_last_token = int(proposed_online_tokens[0, -1])
            should_settle = (
                next_online_len >= args.gamma1
                or proposed_last_token == eos_id
                or next_total_generated >= args.num_max_token
            )

            pending_online_tokens = torch.cat([pending_online_tokens, proposed_online_tokens], dim=1)
            current_token = proposed_online_tokens[:, -1:]

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
                engine.revert_to("final_verify", chunk_start_final_snapshot)

                if authoritative_tokens.numel() == 0:
                    break

                final_prefix = authoritative_tokens[:, :-1]
                engine.commit_prefix("final_verify", chunk_start_final_token, final_prefix)
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
                    f"[Stage Final Verify] draft_elapsed={elapsed_draft:.4f}s final_elapsed={elapsed_final:.4f}s "
                    f"accepted={authoritative_tokens.shape[1]} rejected={final_rejected} unsettled={unsettled_total} "
                    f"budget=1.0 text={to_text(tokenizer, final_outputs)}"
                )

                pending_online_tokens = empty_like_tokens(current_token)
                chunk_start_token = current_token.clone()
                chunk_start_final_token = final_current_token.clone()
                chunk_start_draft_snapshot = engine.snapshot_state("draft")
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
        print(f"Dynamic budget enabled: {args.enable_dynamic_budget} (ignored in 2-stage)")
        print(f"Dynamic route counters: skip={dynamic_skip_count}, high={dynamic_high_count}, normal={dynamic_normal_count}")
        print(f"Speculate calls: {step_speculate_calls}")
        print(f"Early Verify calls: {step_early_verify_calls}")
        print(f"Final Verify calls: {step_final_verify_calls}")
        print(f"Final Verify pre-run calls: {final_prerun_calls}")
        print(f"Tokens generated: {step_tokens_generated}")
        print(f"Drafted tokens: {drafted_tokens_count}")
        print(f"Final settled tokens: {final_settled_tokens_count}")
        print(f"Final authoritative bonus tokens: {final_authoritative_bonus_count}")
        print(f"Emitted tokens: {step_tokens_generated}")
        print(f"Draft cache state: {engine.cache_state_report('draft')}")
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
