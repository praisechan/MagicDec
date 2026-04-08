import os

import torch

from probe_utils import (
    append_rows,
    build_engine,
    build_verified_commit,
    collect_stage_with_hidden,
    empty_like_tokens,
    first_mismatch_idx,
    init_stage_caches,
    parse_common_args,
    replay_verified_prefix,
    softmax_features,
)


def main():
    args = parse_common_args("Phase 2d budget sensitivity probe")
    engine, dataset, prompt_format = build_engine(args)
    tokenizer = engine.model.tokenizer
    eos_id = tokenizer.eos_token_id
    output_csv = args.output_csv or os.path.join(
        os.path.dirname(__file__),
        "data",
        f"phase2d_budget_sensitivity_{args.model_name.replace('/', '_')}_{args.prefix_len}.csv",
    )

    rows = []
    budget_specs = [
        ("budget_lo", max(args.budget2 * 0.8, 1e-4)),
        ("budget_mid", args.budget2),
        ("budget_hi", min(args.budget2 * 1.2, 1.0)),
    ]

    for step in range(min(args.num_eval_steps, len(dataset))):
        batch = dataset[step]
        input_ids = engine.preprocess_input(batch, prompt_format, args.dataset, args.prefix_len)
        init_stage_caches(engine, input_ids, engine.attention_masks, args)
        for cache_name, budget in budget_specs:
            engine._init_single_cache(
                cache_name=cache_name,
                attention_type="RetroInfer",
                retrieval_budget=budget,
                estimation_budget=args.estimate_ratio,
                max_new_tokens=args.num_max_token + args.gamma2 + args.gamma1 + 8,
            )

        current_token = engine.encode(input_ids=input_ids)
        final_current_token = engine.prefill_tokens["final_verify"].clone()
        chunk_start_token = current_token.clone()
        chunk_start_final_token = final_current_token.clone()
        chunk_start_draft_snapshot = engine.snapshot_state("draft")
        chunk_start_early_snapshot = engine.snapshot_state("early_verify")
        chunk_start_early_high_snapshot = engine.snapshot_state("early_verify_high")
        chunk_start_final_snapshot = engine.snapshot_state("final_verify")
        pending_online_tokens = empty_like_tokens(current_token)
        step_tokens_generated = 0
        cycle_idx = 0

        while step_tokens_generated < args.num_max_token and cycle_idx < args.max_cycles:
            draft_snapshot = engine.snapshot_state("draft")
            draft_tokens, _, _ = collect_stage_with_hidden(engine, "draft", current_token, args.gamma1)
            early_snapshot = engine.snapshot_state("early_verify")
            early_high_snapshot = engine.snapshot_state("early_verify_high")
            early_outputs, _, _ = collect_stage_with_hidden(
                engine, "early_verify", current_token, draft_tokens.shape[1] + 1, forced_inputs=draft_tokens
            )
            _accepted_len, _rejected_len, committed_online = build_verified_commit(draft_tokens, early_outputs, eos_id)
            replay_verified_prefix(engine, draft_snapshot, early_snapshot, early_high_snapshot, current_token, committed_online)
            pending_online_tokens = torch.cat([pending_online_tokens, committed_online], dim=1)
            current_token = committed_online[:, -1:]
            should_settle = pending_online_tokens.shape[1] >= args.gamma2 or step_tokens_generated + pending_online_tokens.shape[1] >= args.num_max_token
            if not should_settle:
                continue

            online_span = pending_online_tokens[:, :(args.num_max_token - step_tokens_generated)]
            final_outputs, _final_logits, _ = collect_stage_with_hidden(
                engine, "final_verify", final_current_token, online_span.shape[1] + 1, forced_inputs=online_span
            )
            mismatch = first_mismatch_idx(online_span, final_outputs[:, :online_span.shape[1]])
            cycle_idx += 1

            budget_results = {}
            for cache_name, budget in budget_specs:
                engine.revert_to(cache_name, chunk_start_early_snapshot)
                outputs, logits, _ = collect_stage_with_hidden(
                    engine, cache_name, chunk_start_token, online_span.shape[1] + 1, forced_inputs=online_span
                )
                feats = softmax_features(logits)
                budget_results[cache_name] = (budget, outputs, feats)

            for pos in range(online_span.shape[1] + 1):
                argmax_ids = {}
                margins = {}
                entropies = {}
                for cache_name, (budget, outputs, feats) in budget_results.items():
                    argmax_ids[cache_name] = int(outputs[0, pos])
                    margins[cache_name] = float(feats["margin"][0, pos].item())
                    entropies[cache_name] = float(feats["entropy"][0, pos].item())
                rows.append(
                    {
                        "step": step,
                        "cycle_idx": cycle_idx,
                        "position": pos,
                        "is_bonus_position": int(pos == online_span.shape[1]),
                        "mismatch_position": mismatch,
                        "block_rejected": int(mismatch >= 0),
                        "is_rejected_position": int(mismatch >= 0 and pos == mismatch),
                        "final_argmax_id": int(final_outputs[0, pos]),
                        "budget_lo": budget_specs[0][1],
                        "budget_mid": budget_specs[1][1],
                        "budget_hi": budget_specs[2][1],
                        "argmax_lo": argmax_ids["budget_lo"],
                        "argmax_mid": argmax_ids["budget_mid"],
                        "argmax_hi": argmax_ids["budget_hi"],
                        "argmax_changed_across_budgets": int(len(set(argmax_ids.values())) > 1),
                        "margin_lo": margins["budget_lo"],
                        "margin_mid": margins["budget_mid"],
                        "margin_hi": margins["budget_hi"],
                        "margin_range": max(margins.values()) - min(margins.values()),
                        "entropy_lo": entropies["budget_lo"],
                        "entropy_mid": entropies["budget_mid"],
                        "entropy_hi": entropies["budget_hi"],
                        "entropy_range": max(entropies.values()) - min(entropies.values()),
                        "final_matches_lo": int(argmax_ids["budget_lo"] == int(final_outputs[0, pos])),
                        "final_matches_mid": int(argmax_ids["budget_mid"] == int(final_outputs[0, pos])),
                        "final_matches_hi": int(argmax_ids["budget_hi"] == int(final_outputs[0, pos])),
                    }
                )

            accepted_len = online_span.shape[1] if mismatch < 0 else mismatch
            authoritative_tokens = torch.cat([online_span[:, :accepted_len], final_outputs[:, accepted_len:accepted_len + 1]], dim=1)
            authoritative_tokens = authoritative_tokens[:, :(args.num_max_token - step_tokens_generated)]
            engine.revert_to("draft", chunk_start_draft_snapshot)
            engine.revert_to("early_verify", chunk_start_early_snapshot)
            engine.revert_to("early_verify_high", chunk_start_early_high_snapshot)
            engine.revert_to("final_verify", chunk_start_final_snapshot)
            final_prefix = authoritative_tokens[:, :-1]
            engine.commit_prefix("final_verify", chunk_start_final_token, final_prefix)
            engine.commit_prefix("early_verify", chunk_start_token, final_prefix, sync_source="final_verify", source_replayed=True)
            engine.commit_prefix("early_verify_high", chunk_start_token, final_prefix, sync_source="final_verify", source_replayed=True)
            engine.commit_prefix("draft", chunk_start_token, final_prefix, sync_source="final_verify", source_replayed=True)
            current_token = authoritative_tokens[:, -1:]
            final_current_token = authoritative_tokens[:, -1:]
            step_tokens_generated += authoritative_tokens.shape[1]
            pending_online_tokens = empty_like_tokens(current_token)
            chunk_start_token = current_token.clone()
            chunk_start_final_token = final_current_token.clone()
            chunk_start_draft_snapshot = engine.snapshot_state("draft")
            chunk_start_early_snapshot = engine.snapshot_state("early_verify")
            chunk_start_early_high_snapshot = engine.snapshot_state("early_verify_high")
            chunk_start_final_snapshot = engine.snapshot_state("final_verify")

        engine.clear_kv()

    fieldnames = list(rows[0].keys()) if rows else ["step", "cycle_idx", "position"]
    append_rows(output_csv, fieldnames, rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
