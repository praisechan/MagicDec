import json
import math
import os
import re
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

RECOMMENDED_DRAFT_RULE = "min_margin_low_count"
RECOMMENDED_ASSISTED_RULE = "normal_accept_ratio"


def sanitize_tag(value):
    text = str(value)
    text = text.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "-", text)


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


def load_pg19_dataset():
    try:
        return load_dataset("emozilla/pg19", split="test")
    except Exception as exc:
        fallback_arrow = os.environ.get(
            "MAGICDEC_PG19_TEST_ARROW",
            "/data/hf_cache/datasets/emozilla___pg19/default/0.0.0/c021754c8e01c5b1cc83a1f549c1f97fbbb756b8/pg19-test.arrow",
        )
        if os.path.exists(fallback_arrow):
            print(f"Falling back to cached PG19 arrow: {fallback_arrow} ({exc})")
            return Dataset.from_file(fallback_arrow)
        raise


def get_pg19_prompt_format():
    return (
        "You are given a passage from a book. Read the passage carefully.\n\n"
        "Passage:\n{text}\n\n"
        "Now, continue the text naturally and coherently based on the passage above.\n\n"
        "Continuation:"
    )


def first_mismatch_idx(lhs, rhs):
    max_len = min(lhs.shape[1], rhs.shape[1])
    for idx in range(max_len):
        if int(lhs[0, idx]) != int(rhs[0, idx]):
            return idx
    return -1


def first_eos_idx(tokens, eos_id):
    for idx in range(tokens.shape[1]):
        if int(tokens[0, idx]) == eos_id:
            return idx
    return -1


def empty_like_tokens(ref_token):
    return torch.empty((ref_token.shape[0], 0), dtype=ref_token.dtype, device=ref_token.device)


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


def replay_verified_prefix(
    engine,
    draft_snapshot,
    early_snapshot,
    early_high_snapshot,
    verify_start_token,
    committed_online,
):
    engine.revert_to("draft", draft_snapshot)
    engine.revert_to("early_verify", early_snapshot)
    engine.revert_to("early_verify_high", early_high_snapshot)

    accepted_prefix = committed_online[:, :-1]
    engine.commit_prefix("draft", verify_start_token, accepted_prefix, prefer_draft_replay=True)
    engine.commit_prefix("early_verify", verify_start_token, accepted_prefix)
    engine.commit_prefix("early_verify_high", verify_start_token, accepted_prefix)


def reset_skip_buffer(current_token):
    return None, None, empty_like_tokens(current_token), 0, 0


def tensor_to_float_list(tensor):
    if tensor.numel() == 0:
        return []
    return [float(value) for value in tensor[0].detach().cpu().tolist()]


def safe_mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    mean_value = safe_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def linear_slope(values):
    if len(values) <= 1:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = safe_mean(values)
    numerator = 0.0
    denominator = 0.0
    for idx, value in enumerate(values):
        x_delta = idx - x_mean
        numerator += x_delta * (value - y_mean)
        denominator += x_delta * x_delta
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def summarize_draft_features(feature_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    top1_probs = tensor_to_float_list(feature_dict["top1_prob"])
    top2_probs = tensor_to_float_list(feature_dict["top2_prob"])
    margins = tensor_to_float_list(feature_dict["margin"])
    entropies = tensor_to_float_list(feature_dict["entropy"])

    block_len = len(margins)
    if block_len == 0:
        return {
            "block_len": 0,
            "block_min_margin": 1.0,
            "block_mean_margin": 1.0,
            "block_max_margin": 1.0,
            "block_std_margin": 0.0,
            "block_min_entropy": 0.0,
            "block_mean_entropy": 0.0,
            "block_max_entropy": 0.0,
            "block_std_entropy": 0.0,
            "block_min_margin_pos": -1,
            "block_min_margin_pos_norm": 1.0,
            "block_first_margin": 1.0,
            "block_last_margin": 1.0,
            "block_first_entropy": 0.0,
            "block_last_entropy": 0.0,
            "block_margin_delta": 0.0,
            "block_entropy_delta": 0.0,
            "block_margin_slope": 0.0,
            "block_entropy_slope": 0.0,
            "block_top1_prob_mean": 0.0,
            "block_top2_prob_mean": 0.0,
            "block_top1_prob_min": 0.0,
            "block_top2_prob_max": 0.0,
            "block_low_margin_count_0p05": 0,
            "block_low_margin_count_0p10": 0,
            "block_low_margin_frac_0p05": 0.0,
            "block_low_margin_frac_0p10": 0.0,
            "margin_values": [],
            "entropy_values": [],
            "top1_prob_values": [],
            "top2_prob_values": [],
        }

    min_margin_pos = min(range(block_len), key=lambda idx: margins[idx])
    low_margin_count_0p05 = sum(value < 0.05 for value in margins)
    low_margin_count_0p10 = sum(value < 0.10 for value in margins)

    return {
        "block_len": block_len,
        "block_min_margin": min(margins),
        "block_mean_margin": safe_mean(margins),
        "block_max_margin": max(margins),
        "block_std_margin": safe_std(margins),
        "block_min_entropy": min(entropies),
        "block_mean_entropy": safe_mean(entropies),
        "block_max_entropy": max(entropies),
        "block_std_entropy": safe_std(entropies),
        "block_min_margin_pos": min_margin_pos,
        "block_min_margin_pos_norm": min_margin_pos / max(block_len - 1, 1),
        "block_first_margin": margins[0],
        "block_last_margin": margins[-1],
        "block_first_entropy": entropies[0],
        "block_last_entropy": entropies[-1],
        "block_margin_delta": margins[-1] - margins[0],
        "block_entropy_delta": entropies[-1] - entropies[0],
        "block_margin_slope": linear_slope(margins),
        "block_entropy_slope": linear_slope(entropies),
        "block_top1_prob_mean": safe_mean(top1_probs),
        "block_top2_prob_mean": safe_mean(top2_probs),
        "block_top1_prob_min": min(top1_probs),
        "block_top2_prob_max": max(top2_probs),
        "block_low_margin_count_0p05": low_margin_count_0p05,
        "block_low_margin_count_0p10": low_margin_count_0p10,
        "block_low_margin_frac_0p05": low_margin_count_0p05 / block_len,
        "block_low_margin_frac_0p10": low_margin_count_0p10 / block_len,
        "margin_values": margins,
        "entropy_values": entropies,
        "top1_prob_values": top1_probs,
        "top2_prob_values": top2_probs,
    }


def compute_risk_score(block_features, rule_name):
    min_margin = float(block_features["block_min_margin"])
    mean_margin = float(block_features["block_mean_margin"])
    last_margin = float(block_features["block_last_margin"])
    early_min_weight = 1.0 - float(block_features["block_min_margin_pos_norm"])
    low_margin_frac = float(block_features["block_low_margin_frac_0p10"])

    if rule_name in {"baseline_min_gap", "min_gap"}:
        return 1.0 - min_margin
    if rule_name == "mean_margin":
        return 1.0 - mean_margin
    if rule_name == "last_margin":
        return 1.0 - last_margin
    if rule_name == "min_margin_mean_margin":
        return (1.0 - min_margin) + 0.5 * (1.0 - mean_margin)
    if rule_name == "min_margin_low_count":
        return (1.0 - min_margin) + 0.5 * low_margin_frac
    if rule_name == "min_margin_early_position":
        return (1.0 - min_margin) + 0.35 * early_min_weight
    if rule_name == "min_margin_mean_early":
        return (1.0 - min_margin) + 0.4 * (1.0 - mean_margin) + 0.2 * early_min_weight
    raise KeyError(f"Unknown draft risk rule: {rule_name}")


def compute_assisted_risk_score(block_features, rule_name):
    early_accept_ratio = float(block_features.get("normal_accept_ratio", 1.0))
    early_min_margin = float(block_features["block_min_margin"])

    if rule_name == "normal_accept_ratio":
        return 1.0 - early_accept_ratio
    if rule_name == "normal_accept_plus_min_margin":
        return (1.0 - early_accept_ratio) + 0.5 * (1.0 - early_min_margin)
    raise KeyError(f"Unknown assisted risk rule: {rule_name}")


def compute_cost_breakdown(
    budget1,
    budget2,
    budget2_high,
    draft_calls,
    early_normal_calls,
    early_high_calls,
    final_calls,
):
    draft_cost = budget1 * draft_calls
    early_normal_cost = budget2 * early_normal_calls
    early_high_cost = budget2_high * early_high_calls
    final_cost = 1.0 * final_calls
    total_cost = draft_cost + early_normal_cost + early_high_cost + final_cost
    return {
        "draft_cost": draft_cost,
        "early_normal_cost": early_normal_cost,
        "early_high_cost": early_high_cost,
        "final_cost": final_cost,
        "total_cost": total_cost,
    }


def compute_decode_weighted_cost_breakdown(
    budget1,
    budget2,
    budget2_high,
    draft_decode_calls,
    early_normal_decode_calls,
    early_high_decode_calls,
    final_decode_calls,
):
    draft_cost = budget1 * draft_decode_calls
    early_normal_cost = budget2 * early_normal_decode_calls
    early_high_cost = budget2_high * early_high_decode_calls
    final_cost = 1.0 * final_decode_calls
    total_cost = draft_cost + early_normal_cost + early_high_cost + final_cost
    return {
        "decode_weighted_draft_cost": draft_cost,
        "decode_weighted_early_normal_cost": early_normal_cost,
        "decode_weighted_early_high_cost": early_high_cost,
        "decode_weighted_final_cost": final_cost,
        "decode_weighted_total_cost": total_cost,
    }
