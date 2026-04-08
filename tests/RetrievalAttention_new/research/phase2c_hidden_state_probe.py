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
)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main():
    args = parse_common_args("Phase 2c hidden-state probe")
    engine, dataset, prompt_format = build_engine(args)
    tokenizer = engine.model.tokenizer
    eos_id = tokenizer.eos_token_id
    output_csv = args.output_csv or os.path.join(
        os.path.dirname(__file__),
        "data",
        f"phase2c_hidden_state_probe_{args.model_name.replace('/', '_')}_{args.prefix_len}.csv",
    )

    rows = []
    for step in range(min(args.num_eval_steps, len(dataset))):
        batch = dataset[step]
        input_ids = engine.preprocess_input(batch, prompt_format, args.dataset, args.prefix_len)
        init_stage_caches(engine, input_ids, engine.attention_masks, args)

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
            early_outputs, _early_logits, early_hidden = collect_stage_with_hidden(
                engine, "early_verify", current_token, draft_tokens.shape[1] + 1, forced_inputs=draft_tokens
            )
            _accepted_len, _rejected_len, committed_online = build_verified_commit(draft_tokens, early_outputs, eos_id)
            replay_verified_prefix(engine, draft_snapshot=draft_snapshot, early_snapshot=early_snapshot, early_high_snapshot=early_high_snapshot, verify_start_token=current_token, committed_online=committed_online)

            pending_online_tokens = torch.cat([pending_online_tokens, committed_online], dim=1)
            current_token = committed_online[:, -1:]
            should_settle = pending_online_tokens.shape[1] >= args.gamma2 or step_tokens_generated + pending_online_tokens.shape[1] >= args.num_max_token
            if not should_settle:
                continue

            online_span = pending_online_tokens[:, :(args.num_max_token - step_tokens_generated)]
            final_outputs, _final_logits, final_hidden = collect_stage_with_hidden(
                engine, "final_verify", final_current_token, online_span.shape[1] + 1, forced_inputs=online_span
            )
            mismatch = first_mismatch_idx(online_span, final_outputs[:, :online_span.shape[1]])
            cycle_idx += 1

            for pos in range(online_span.shape[1] + 1):
                if pos >= early_hidden.shape[1]:
                    continue
                early_vec = early_hidden[0, pos]
                final_vec = final_hidden[0, pos]
                diff = final_vec - early_vec
                rows.append(
                    {
                        "step": step,
                        "cycle_idx": cycle_idx,
                        "position": pos,
                        "is_bonus_position": int(pos == online_span.shape[1]),
                        "mismatch_position": mismatch,
                        "rel_to_mismatch": pos - mismatch if mismatch >= 0 else "",
                        "block_rejected": int(mismatch >= 0),
                        "is_rejected_position": int(mismatch >= 0 and pos == mismatch),
                        "cosine_early_final_hidden": cosine_similarity(early_vec, final_vec),
                        "l2_hidden_diff": float(torch.norm(diff, p=2).item()),
                        "linf_hidden_diff": float(torch.norm(diff, p=float("inf")).item()),
                        "early_hidden_norm": float(torch.norm(early_vec, p=2).item()),
                        "final_hidden_norm": float(torch.norm(final_vec, p=2).item()),
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
