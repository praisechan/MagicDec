import os

from probe_utils import (
    append_rows,
    build_engine,
    build_verified_commit,
    collect_stage_with_hidden,
    empty_like_tokens,
    encode_tensor_payload,
    first_eos_idx,
    first_mismatch_idx,
    init_stage_caches,
    js_divergence,
    kl_divergence,
    parse_common_args,
    replay_verified_prefix,
    reset_skip_buffer,
    softmax_features,
    topk_overlap_and_rank,
    total_variation_distance,
)


def main():
    args = parse_common_args("Phase 2b logit-level probe")
    engine, dataset, prompt_format = build_engine(args)
    tokenizer = engine.model.tokenizer
    eos_id = tokenizer.eos_token_id

    output_csv = args.output_csv or os.path.join(
        os.path.dirname(__file__),
        "data",
        f"phase2b_logit_probe_{args.model_name.replace('/', '_')}_{args.prefix_len}.csv",
    )

    rows = []
    cycle_count = 0
    for step in range(min(args.num_eval_steps, len(dataset))):
        batch = dataset[step]
        input_ids = engine.preprocess_input(batch, prompt_format, args.dataset, args.prefix_len)
        attention_masks = engine.attention_masks
        init_stage_caches(engine, input_ids, attention_masks, args)

        current_token = engine.encode(input_ids=input_ids)
        final_current_token = engine.prefill_tokens["final_verify"].clone()
        pending_online_tokens = empty_like_tokens(current_token)
        chunk_start_token = current_token.clone()
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
        step_tokens_generated = 0

        while step_tokens_generated < args.num_max_token and cycle_count < args.max_cycles:
            if consecutive_skip_count > 0:
                draft_decode_len = args.gamma1 * (consecutive_skip_count + 1)
                draft_start_token = skip_anchor_token
                engine.revert_to("draft", skip_anchor_draft_snapshot)
                draft_snapshot = skip_anchor_draft_snapshot
            else:
                draft_decode_len = args.gamma1
                draft_start_token = current_token
                draft_snapshot = engine.snapshot_state("draft")

            draft_tokens, draft_logits, _draft_hidden = collect_stage_with_hidden(
                engine, "draft", draft_start_token, draft_decode_len
            )
            draft_probs = softmax_features(draft_logits)
            min_conf = float(draft_probs["margin"].min().item())

            dynamic_mode = "normal"
            if args.enable_dynamic_budget:
                if min_conf > args.T_high:
                    dynamic_mode = "skip"
                elif min_conf < args.T_low:
                    dynamic_mode = "high"

            if dynamic_mode == "skip":
                if consecutive_skip_count == 0:
                    skip_anchor_token = current_token.clone()
                    skip_anchor_draft_snapshot = draft_snapshot
                eos_idx = first_eos_idx(draft_tokens, eos_id)
                accepted_len = (eos_idx + 1) if eos_idx >= 0 else draft_tokens.shape[1]
                skip_stacked_draft_tokens = draft_tokens[:, :accepted_len]
                consecutive_skip_count += 1
                engine.revert_to("draft", draft_snapshot)

                skip_next_online_len = pending_online_tokens.shape[1] + skip_stacked_draft_tokens.shape[1]
                skip_next_total_generated = step_tokens_generated + skip_next_online_len
                skip_last_token = int(skip_stacked_draft_tokens[0, -1])
                should_force_skip_safety_verify = (
                    skip_next_online_len >= args.gamma2
                    or skip_last_token == eos_id
                    or skip_next_total_generated >= args.num_max_token
                )
                if not should_force_skip_safety_verify:
                    continue

                verify_start_token = skip_anchor_token
                verify_tokens = skip_stacked_draft_tokens
                verify_mode = "normal"
                early_snapshot = engine.snapshot_state("early_verify")
                early_high_snapshot = engine.snapshot_state("early_verify_high")
                early_cache_name = "early_verify"
                early_outputs, early_logits, _early_hidden = collect_stage_with_hidden(
                    engine, early_cache_name, verify_start_token, verify_tokens.shape[1] + 1, forced_inputs=verify_tokens
                )
                accepted_len, rejected_len, committed_online = build_verified_commit(verify_tokens, early_outputs, eos_id)
                replay_verified_prefix(
                    engine,
                    skip_anchor_draft_snapshot,
                    early_snapshot,
                    early_high_snapshot,
                    verify_start_token,
                    committed_online,
                )
                skip_anchor_token, skip_anchor_draft_snapshot, skip_stacked_draft_tokens, consecutive_skip_count, skip_accounted_draft_len = reset_skip_buffer(current_token)
                proposed_online_tokens = committed_online
            else:
                verify_start_token = skip_anchor_token if consecutive_skip_count > 0 else current_token
                verify_tokens = draft_tokens
                verify_mode = dynamic_mode
                early_snapshot = engine.snapshot_state("early_verify")
                early_high_snapshot = engine.snapshot_state("early_verify_high")
                early_cache_name = "early_verify_high" if dynamic_mode == "high" else "early_verify"
                early_outputs, early_logits, _early_hidden = collect_stage_with_hidden(
                    engine, early_cache_name, verify_start_token, verify_tokens.shape[1] + 1, forced_inputs=verify_tokens
                )
                accepted_len, rejected_len, committed_online = build_verified_commit(verify_tokens, early_outputs, eos_id)
                replay_verified_prefix(
                    engine,
                    skip_anchor_draft_snapshot if consecutive_skip_count > 0 else draft_snapshot,
                    early_snapshot,
                    early_high_snapshot,
                    verify_start_token,
                    committed_online,
                )
                skip_anchor_token, skip_anchor_draft_snapshot, skip_stacked_draft_tokens, consecutive_skip_count, skip_accounted_draft_len = reset_skip_buffer(current_token)
                proposed_online_tokens = committed_online

            if proposed_online_tokens is None or proposed_online_tokens.numel() == 0:
                break

            pending_online_tokens = torch.cat([pending_online_tokens, proposed_online_tokens], dim=1)
            current_token = proposed_online_tokens[:, -1:]
            should_settle = (
                pending_online_tokens.shape[1] >= args.gamma2
                or int(proposed_online_tokens[0, -1]) == eos_id
                or (step_tokens_generated + pending_online_tokens.shape[1]) >= args.num_max_token
            )
            if not should_settle:
                continue

            online_span = pending_online_tokens[:, :(args.num_max_token - step_tokens_generated)]
            final_outputs, final_logits, _final_hidden = collect_stage_with_hidden(
                engine, "final_verify", final_current_token, online_span.shape[1] + 1, forced_inputs=online_span
            )
            mismatch = first_mismatch_idx(online_span, final_outputs[:, :online_span.shape[1]])
            final_probs = softmax_features(final_logits)
            early_probs = softmax_features(early_logits)

            mismatch_pos = mismatch
            cycle_count += 1
            for pos in range(online_span.shape[1] + 1):
                is_bonus = int(pos == online_span.shape[1])
                is_rejected_pos = int(mismatch_pos >= 0 and pos == mismatch_pos)
                rel_to_mismatch = pos - mismatch_pos if mismatch_pos >= 0 else ""
                final_prob_vec = final_probs["probs"][0, pos]
                has_early = pos < early_logits.shape[1]
                has_draft = pos < draft_logits.shape[1]
                early_prob_vec = early_probs["probs"][0, pos] if has_early else None
                draft_prob_vec = draft_probs["probs"][0, pos] if has_draft else None
                overlap10, rank_in_early = ("", "")
                if has_early:
                    overlap10, rank_in_early = topk_overlap_and_rank(final_prob_vec, early_prob_vec, k=10)
                rows.append(
                    {
                        "step": step,
                        "cycle_idx": cycle_count,
                        "verify_mode": verify_mode,
                        "position": pos,
                        "is_bonus_position": is_bonus,
                        "mismatch_position": mismatch_pos,
                        "rel_to_mismatch": rel_to_mismatch,
                        "block_rejected": int(mismatch_pos >= 0),
                        "is_rejected_position": is_rejected_pos,
                        "draft_argmax_id": int(draft_tokens[0, pos]) if has_draft else "",
                        "early_argmax_id": int(early_outputs[0, pos]) if has_early else "",
                        "final_argmax_id": int(final_outputs[0, pos]),
                        "final_top1_rank_in_early": rank_in_early,
                        "top10_overlap_early_final": overlap10,
                        "kl_final_early": float(kl_divergence(final_prob_vec, early_prob_vec).item()) if has_early else "",
                        "kl_final_draft": float(kl_divergence(final_prob_vec, draft_prob_vec).item()) if has_draft else "",
                        "js_final_early": float(js_divergence(final_prob_vec, early_prob_vec).item()) if has_early else "",
                        "tv_final_early": float(total_variation_distance(final_prob_vec, early_prob_vec).item()) if has_early else "",
                        "draft_margin": float(draft_probs["margin"][0, pos].item()) if has_draft else "",
                        "early_margin": float(early_probs["margin"][0, pos].item()) if has_early else "",
                        "final_margin": float(final_probs["margin"][0, pos].item()),
                        "draft_entropy": float(draft_probs["entropy"][0, pos].item()) if has_draft else "",
                        "early_entropy": float(early_probs["entropy"][0, pos].item()) if has_early else "",
                        "final_entropy": float(final_probs["entropy"][0, pos].item()),
                        "draft_logits_payload": encode_tensor_payload(draft_logits[0, pos]) if has_draft else "",
                        "early_logits_payload": encode_tensor_payload(early_logits[0, pos]) if has_early else "",
                        "final_logits_payload": encode_tensor_payload(final_logits[0, pos]),
                        "draft_probs_payload": encode_tensor_payload(draft_prob_vec) if has_draft else "",
                        "early_probs_payload": encode_tensor_payload(early_prob_vec) if has_early else "",
                        "final_probs_payload": encode_tensor_payload(final_prob_vec),
                    }
                )

            accepted_len = online_span.shape[1] if mismatch < 0 else mismatch
            authoritative_tokens = torch.cat(
                [online_span[:, :accepted_len], final_outputs[:, accepted_len:accepted_len + 1]],
                dim=1,
            )
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

    fieldnames = list(rows[0].keys()) if rows else [
        "step", "cycle_idx", "verify_mode", "position", "is_bonus_position", "mismatch_position",
    ]
    append_rows(output_csv, fieldnames, rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    import torch
    main()
