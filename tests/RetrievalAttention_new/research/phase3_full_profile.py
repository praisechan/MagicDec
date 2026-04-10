import json
import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from probe_utils import (
    append_rows,
    build_engine,
    build_verified_commit,
    collect_stage_with_hidden,
    empty_like_tokens,
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


def topk_triplet(logits: torch.Tensor, probs: torch.Tensor, k: int = 10):
    top = torch.topk(probs, k=k, dim=-1)
    ids = top.indices.detach().cpu().tolist()
    top_logits = logits[top.indices].detach().cpu().tolist()
    top_probs = top.values.detach().cpu().tolist()
    return json.dumps(ids), json.dumps(top_logits), json.dumps(top_probs)


def scalar_item(value: Optional[torch.Tensor]):
    if value is None:
        return ""
    return float(value.item())


def metric_if_available(fn, lhs: Optional[torch.Tensor], rhs: Optional[torch.Tensor]):
    if lhs is None or rhs is None:
        return ""
    return float(fn(lhs, rhs).item())


def overlap_if_available(lhs: Optional[torch.Tensor], rhs: Optional[torch.Tensor], k: int = 10):
    if lhs is None or rhs is None:
        return "", ""
    return topk_overlap_and_rank(lhs, rhs, k=k)


def build_materialized_entries(
    committed_online: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_probs: Dict[str, torch.Tensor],
    early_outputs: torch.Tensor,
    early_logits: torch.Tensor,
    early_probs: Dict[str, torch.Tensor],
    accepted_len: int,
    verify_mode: str,
    materialization_idx: int,
) -> List[Dict[str, object]]:
    entries = []
    for local_pos in range(committed_online.shape[1]):
        is_accepted_prefix = local_pos < accepted_len
        draft_prob_vec = draft_probs["probs"][0, local_pos].detach().cpu() if is_accepted_prefix else None
        draft_logit_vec = draft_logits[0, local_pos].detach().cpu() if is_accepted_prefix else None
        early_prob_vec = early_probs["probs"][0, local_pos].detach().cpu()
        early_logit_vec = early_logits[0, local_pos].detach().cpu()
        entries.append(
            {
                "materialization_idx": materialization_idx,
                "materialized_local_pos": local_pos,
                "source_verify_mode": verify_mode,
                "source_kind": "accepted_prefix" if is_accepted_prefix else "early_bonus",
                "source_local_pos": local_pos if is_accepted_prefix else accepted_len,
                "source_is_bonus_from_early": int(not is_accepted_prefix),
                "source_accepted_len": accepted_len,
                "online_token_id": int(committed_online[0, local_pos]),
                "draft_token_id": int(draft_tokens[0, local_pos]) if is_accepted_prefix else "",
                "early_token_id": int(early_outputs[0, local_pos]),
                "draft_prob_vec": draft_prob_vec,
                "draft_logit_vec": draft_logit_vec,
                "early_prob_vec": early_prob_vec,
                "early_logit_vec": early_logit_vec,
                "draft_margin": scalar_item(draft_probs["margin"][0, local_pos]) if is_accepted_prefix else "",
                "draft_entropy": scalar_item(draft_probs["entropy"][0, local_pos]) if is_accepted_prefix else "",
                "early_margin": scalar_item(early_probs["margin"][0, local_pos]),
                "early_entropy": scalar_item(early_probs["entropy"][0, local_pos]),
            }
        )
    return entries


def finalize_rows(
    step: int,
    cycle_idx: int,
    pending_entries: List[Dict[str, object]],
    online_span: torch.Tensor,
    final_outputs: torch.Tensor,
    final_logits: torch.Tensor,
    mismatch_pos: int,
) -> List[Dict[str, object]]:
    final_probs = softmax_features(final_logits)
    rows = []
    for pos in range(online_span.shape[1] + 1):
        pending_entry = pending_entries[pos] if pos < len(pending_entries) else None
        draft_prob_vec = pending_entry["draft_prob_vec"] if pending_entry is not None else None
        draft_logit_vec = pending_entry["draft_logit_vec"] if pending_entry is not None else None
        early_prob_vec = pending_entry["early_prob_vec"] if pending_entry is not None else None
        early_logit_vec = pending_entry["early_logit_vec"] if pending_entry is not None else None
        final_prob_vec = final_probs["probs"][0, pos].detach().cpu()
        final_logit_vec = final_logits[0, pos].detach().cpu()

        draft_top10_ids, draft_top10_logits, draft_top10_probs = (
            topk_triplet(draft_logit_vec, draft_prob_vec) if draft_prob_vec is not None else ("", "", "")
        )
        early_top10_ids, early_top10_logits, early_top10_probs = (
            topk_triplet(early_logit_vec, early_prob_vec) if early_prob_vec is not None else ("", "", "")
        )
        final_top10_ids, final_top10_logits, final_top10_probs = topk_triplet(final_logit_vec, final_prob_vec)

        early_final_overlap10, final_top1_rank_in_early = overlap_if_available(final_prob_vec, early_prob_vec, k=10)
        draft_final_overlap10, final_top1_rank_in_draft = overlap_if_available(final_prob_vec, draft_prob_vec, k=10)
        draft_early_overlap10, early_top1_rank_in_draft = overlap_if_available(early_prob_vec, draft_prob_vec, k=10)

        row = {
            "step": step,
            "cycle_idx": cycle_idx,
            "position": pos,
            "is_bonus_position": int(pos == online_span.shape[1]),
            "mismatch_position": mismatch_pos,
            "first_mismatch_position": mismatch_pos,
            "rel_to_mismatch": pos - mismatch_pos if mismatch_pos >= 0 else "",
            "block_rejected": int(mismatch_pos >= 0),
            "accepted_rejected_flag": "rejected" if mismatch_pos >= 0 else "accepted",
            "is_rejected_position": int(mismatch_pos >= 0 and pos == mismatch_pos),
            "pre_final_features_available": int(pending_entry is not None),
            "online_token_id": int(online_span[0, pos]) if pos < online_span.shape[1] else "",
            "final_argmax_id": int(final_outputs[0, pos]),
            "final_top1_id": int(final_probs["top1_id"][0, pos].item()),
            "final_margin": float(final_probs["margin"][0, pos].item()),
            "final_entropy": float(final_probs["entropy"][0, pos].item()),
            "final_top10_token_ids": final_top10_ids,
            "final_top10_logits": final_top10_logits,
            "final_top10_probs": final_top10_probs,
        }
        if pending_entry is not None:
            row.update(
                {
                    "materialization_idx": pending_entry["materialization_idx"],
                    "materialized_local_pos": pending_entry["materialized_local_pos"],
                    "source_verify_mode": pending_entry["source_verify_mode"],
                    "source_kind": pending_entry["source_kind"],
                    "source_local_pos": pending_entry["source_local_pos"],
                    "source_is_bonus_from_early": pending_entry["source_is_bonus_from_early"],
                    "source_accepted_len": pending_entry["source_accepted_len"],
                    "draft_argmax_id": pending_entry["draft_token_id"],
                    "early_argmax_id": pending_entry["early_token_id"],
                    "draft_top1_id": int(draft_prob_vec.argmax().item()) if draft_prob_vec is not None else "",
                    "early_top1_id": int(early_prob_vec.argmax().item()),
                    "draft_margin": pending_entry["draft_margin"],
                    "early_margin": pending_entry["early_margin"],
                    "draft_entropy": pending_entry["draft_entropy"],
                    "early_entropy": pending_entry["early_entropy"],
                    "draft_top10_token_ids": draft_top10_ids,
                    "draft_top10_logits": draft_top10_logits,
                    "draft_top10_probs": draft_top10_probs,
                    "early_top10_token_ids": early_top10_ids,
                    "early_top10_logits": early_top10_logits,
                    "early_top10_probs": early_top10_probs,
                    "top10_overlap_early_final": early_final_overlap10,
                    "top10_overlap_draft_final": draft_final_overlap10,
                    "top10_overlap_draft_early": draft_early_overlap10,
                    "final_top1_rank_in_early": final_top1_rank_in_early,
                    "final_top1_rank_in_draft": final_top1_rank_in_draft,
                    "early_top1_rank_in_draft": early_top1_rank_in_draft,
                    "kl_final_early": metric_if_available(kl_divergence, final_prob_vec, early_prob_vec),
                    "kl_final_draft": metric_if_available(kl_divergence, final_prob_vec, draft_prob_vec),
                    "kl_early_draft": metric_if_available(kl_divergence, early_prob_vec, draft_prob_vec),
                    "js_final_early": metric_if_available(js_divergence, final_prob_vec, early_prob_vec),
                    "js_final_draft": metric_if_available(js_divergence, final_prob_vec, draft_prob_vec),
                    "js_early_draft": metric_if_available(js_divergence, early_prob_vec, draft_prob_vec),
                    "tv_final_early": metric_if_available(total_variation_distance, final_prob_vec, early_prob_vec),
                    "tv_final_draft": metric_if_available(total_variation_distance, final_prob_vec, draft_prob_vec),
                    "tv_early_draft": metric_if_available(total_variation_distance, early_prob_vec, draft_prob_vec),
                }
            )
        else:
            row.update(
                {
                    "materialization_idx": "",
                    "materialized_local_pos": "",
                    "source_verify_mode": "",
                    "source_kind": "",
                    "source_local_pos": "",
                    "source_is_bonus_from_early": "",
                    "source_accepted_len": "",
                    "draft_argmax_id": "",
                    "early_argmax_id": "",
                    "draft_top1_id": "",
                    "early_top1_id": "",
                    "draft_margin": "",
                    "early_margin": "",
                    "draft_entropy": "",
                    "early_entropy": "",
                    "draft_top10_token_ids": "",
                    "draft_top10_logits": "",
                    "draft_top10_probs": "",
                    "early_top10_token_ids": "",
                    "early_top10_logits": "",
                    "early_top10_probs": "",
                    "top10_overlap_early_final": "",
                    "top10_overlap_draft_final": "",
                    "top10_overlap_draft_early": "",
                    "final_top1_rank_in_early": "",
                    "final_top1_rank_in_draft": "",
                    "early_top1_rank_in_draft": "",
                    "kl_final_early": "",
                    "kl_final_draft": "",
                    "kl_early_draft": "",
                    "js_final_early": "",
                    "js_final_draft": "",
                    "js_early_draft": "",
                    "tv_final_early": "",
                    "tv_final_draft": "",
                    "tv_early_draft": "",
                }
            )
        rows.append(row)
    return rows


def main():
    args = parse_common_args("Phase 3 full aligned profile")
    engine, dataset, prompt_format = build_engine(args)
    eos_id = engine.model.tokenizer.eos_token_id

    output_csv = args.output_csv or os.path.join(
        os.path.dirname(__file__),
        "data",
        f"phase3_full_profile_{args.model_name.replace('/', '_')}_{args.prefix_len}.csv",
    )

    rows = []

    for step in tqdm(range(min(args.num_eval_steps, len(dataset)))):
        cycle_count = 0
        batch = dataset[step]
        input_ids = engine.preprocess_input(batch, prompt_format, args.dataset, args.prefix_len)
        if input_ids.shape[1] > 40000:
            continue
        attention_masks = engine.attention_masks
        init_stage_caches(engine, input_ids, attention_masks, args)

        current_token = engine.encode(input_ids=input_ids)
        final_current_token = engine.prefill_tokens["final_verify"].clone()
        pending_online_tokens = empty_like_tokens(current_token)
        pending_entries: List[Dict[str, object]] = []
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
        materialization_idx = 0

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

            draft_tokens, draft_logits, _ = collect_stage_with_hidden(engine, "draft", draft_start_token, draft_decode_len)
            draft_probs = softmax_features(draft_logits)
            min_conf = float(draft_probs["margin"].min().item())

            dynamic_mode = "normal"
            if args.enable_dynamic_budget:
                if min_conf > args.T_high:
                    dynamic_mode = "skip"
                elif min_conf < args.T_low:
                    dynamic_mode = "high"

            committed_online = None
            materialized_entries = None
            verify_mode = dynamic_mode

            if dynamic_mode == "skip":
                if consecutive_skip_count == 0:
                    skip_anchor_token = current_token.clone()
                    skip_anchor_draft_snapshot = draft_snapshot

                eos_idx = first_eos_idx(draft_tokens, eos_id)
                accepted_len = (eos_idx + 1) if eos_idx >= 0 else draft_tokens.shape[1]
                skip_stacked_draft_tokens = draft_tokens[:, :accepted_len]
                consecutive_skip_count += 1
                engine.revert_to("draft", draft_snapshot)

                if skip_stacked_draft_tokens.numel() == 0:
                    break

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
                early_outputs, early_logits, _ = collect_stage_with_hidden(
                    engine, "early_verify", verify_start_token, verify_tokens.shape[1] + 1, forced_inputs=verify_tokens
                )
                early_probs = softmax_features(early_logits)
                accepted_len, rejected_len, committed_online = build_verified_commit(verify_tokens, early_outputs, eos_id)
                materialized_entries = build_materialized_entries(
                    committed_online=committed_online,
                    draft_tokens=verify_tokens,
                    draft_logits=draft_logits[:, :verify_tokens.shape[1], :],
                    draft_probs={
                        key: value[:, :verify_tokens.shape[1], ...] if isinstance(value, torch.Tensor) and value.dim() >= 2 else value
                        for key, value in draft_probs.items()
                    },
                    early_outputs=early_outputs,
                    early_logits=early_logits,
                    early_probs=early_probs,
                    accepted_len=accepted_len,
                    verify_mode=verify_mode,
                    materialization_idx=materialization_idx,
                )
                replay_verified_prefix(
                    engine,
                    skip_anchor_draft_snapshot,
                    early_snapshot,
                    early_high_snapshot,
                    verify_start_token,
                    committed_online,
                )
                skip_anchor_token, skip_anchor_draft_snapshot, skip_stacked_draft_tokens, consecutive_skip_count, skip_accounted_draft_len = reset_skip_buffer(current_token)
            else:
                verify_start_token = skip_anchor_token if consecutive_skip_count > 0 else current_token
                verify_tokens = draft_tokens
                early_snapshot = engine.snapshot_state("early_verify")
                early_high_snapshot = engine.snapshot_state("early_verify_high")
                early_cache_name = "early_verify_high" if dynamic_mode == "high" else "early_verify"
                early_outputs, early_logits, _ = collect_stage_with_hidden(
                    engine, early_cache_name, verify_start_token, verify_tokens.shape[1] + 1, forced_inputs=verify_tokens
                )
                early_probs = softmax_features(early_logits)
                accepted_len, rejected_len, committed_online = build_verified_commit(verify_tokens, early_outputs, eos_id)
                materialized_entries = build_materialized_entries(
                    committed_online=committed_online,
                    draft_tokens=verify_tokens,
                    draft_logits=draft_logits,
                    draft_probs=draft_probs,
                    early_outputs=early_outputs,
                    early_logits=early_logits,
                    early_probs=early_probs,
                    accepted_len=accepted_len,
                    verify_mode=verify_mode,
                    materialization_idx=materialization_idx,
                )
                replay_verified_prefix(
                    engine,
                    skip_anchor_draft_snapshot if consecutive_skip_count > 0 else draft_snapshot,
                    early_snapshot,
                    early_high_snapshot,
                    verify_start_token,
                    committed_online,
                )
                skip_anchor_token, skip_anchor_draft_snapshot, skip_stacked_draft_tokens, consecutive_skip_count, skip_accounted_draft_len = reset_skip_buffer(current_token)

            if committed_online is None or committed_online.numel() == 0:
                break

            pending_online_tokens = torch.cat([pending_online_tokens, committed_online], dim=1)
            pending_entries.extend(materialized_entries)
            materialization_idx += 1
            current_token = committed_online[:, -1:]
            should_settle = (
                pending_online_tokens.shape[1] >= args.gamma2
                or int(committed_online[0, -1]) == eos_id
                or (step_tokens_generated + pending_online_tokens.shape[1]) >= args.num_max_token
            )
            if not should_settle:
                continue

            online_span = pending_online_tokens[:, :(args.num_max_token - step_tokens_generated)]
            if len(pending_entries) < online_span.shape[1]:
                raise RuntimeError("Phase 3 alignment error: pending feature ledger shorter than online span.")

            final_outputs, final_logits, _ = collect_stage_with_hidden(
                engine, "final_verify", final_current_token, online_span.shape[1] + 1, forced_inputs=online_span
            )
            mismatch = first_mismatch_idx(online_span, final_outputs[:, :online_span.shape[1]])
            cycle_count += 1
            rows.extend(
                finalize_rows(
                    step=step,
                    cycle_idx=cycle_count,
                    pending_entries=pending_entries[:online_span.shape[1]],
                    online_span=online_span,
                    final_outputs=final_outputs,
                    final_logits=final_logits,
                    mismatch_pos=mismatch,
                )
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
            pending_entries = []
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
