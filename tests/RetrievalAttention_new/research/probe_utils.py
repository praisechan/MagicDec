import argparse
import base64
import csv
import json
import os
import sys
import zlib
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
TESTS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_ROOT, "../.."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from MagicDec.Engine.RetrievalAttention_new.backend import LMBackend
from MagicDec.Engine.utils import setup_seed


def parse_common_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3.1-8B")
    parser.add_argument("--dataset", type=str, default="pg19", choices=["pg19", "longbenchv1"])
    parser.add_argument("--task", type=str, default="gov_report")
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--prefix_len", type=int, default=32768)
    parser.add_argument("--gamma1", type=int, default=6)
    parser.add_argument("--gamma2", type=int, default=32)
    parser.add_argument("--budget1", type=float, default=0.02)
    parser.add_argument("--budget2", type=float, default=0.10)
    parser.add_argument("--budget2_high", type=float, default=0.20)
    parser.add_argument("--estimate_ratio", type=float, default=0.25)
    parser.add_argument("--num_max_token", type=int, default=100)
    parser.add_argument("--num_eval_steps", type=int, default=1)
    parser.add_argument("--max_cycles", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--enable_dynamic_budget", action="store_true")
    parser.add_argument("--T_low", type=float, default=0.05)
    parser.add_argument("--T_high", type=float, default=0.20)
    parser.add_argument("--output_csv", type=str, default="")
    return parser.parse_args()


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


def build_engine(args):
    setup_seed(args.seed)
    model2path, model2maxlen, dataset2prompt = load_longbench_config()
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_device = "auto" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = model2path[args.model_name]
    max_length = model2maxlen[args.model_name]
    prompt_format = get_pg19_prompt_format() if args.dataset == "pg19" else dataset2prompt[args.task]

    engine = LMBackend(dtype=torch.bfloat16, device=runtime_device, dec_len=args.gamma1 + 1)
    engine.load_model(model_path, max_length, torch.bfloat16, model_device, args.B)

    if args.dataset == "pg19":
        dataset = load_pg19_dataset()
    else:
        dataset = load_dataset("THUDM/LongBench", args.task, split="test", trust_remote_code=True)

    return engine, dataset, prompt_format


def init_stage_caches(engine, input_ids, attention_masks, args):
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


def empty_like_tokens(ref_token):
    return torch.empty((ref_token.shape[0], 0), dtype=ref_token.dtype, device=ref_token.device)


def first_eos_idx(tokens, eos_id):
    for idx in range(tokens.shape[1]):
        if int(tokens[0, idx]) == eos_id:
            return idx
    return -1


def first_mismatch_idx(lhs, rhs):
    max_len = min(lhs.shape[1], rhs.shape[1])
    for idx in range(max_len):
        if int(lhs[0, idx]) != int(rhs[0, idx]):
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


def replay_verified_prefix(engine, draft_snapshot, early_snapshot, early_high_snapshot, verify_start_token, committed_online):
    engine.revert_to("draft", draft_snapshot)
    engine.revert_to("early_verify", early_snapshot)
    engine.revert_to("early_verify_high", early_high_snapshot)
    accepted_prefix = committed_online[:, :-1]
    engine.commit_prefix("draft", verify_start_token, accepted_prefix, prefer_draft_replay=True)
    engine.commit_prefix("early_verify", verify_start_token, accepted_prefix)
    engine.commit_prefix("early_verify_high", verify_start_token, accepted_prefix)


def reset_skip_buffer(current_token):
    return None, None, empty_like_tokens(current_token), 0, 0


def softmax_features(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    probs = torch.softmax(logits.float(), dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return {
        "probs": probs,
        "top1_prob": top2.values[..., 0],
        "top2_prob": top2.values[..., 1],
        "margin": top2.values[..., 0] - top2.values[..., 1],
        "entropy": entropy,
        "top1_id": top2.indices[..., 0],
        "top2_id": top2.indices[..., 1],
    }


def encode_tensor_payload(tensor: torch.Tensor, dtype: str = "float16") -> str:
    arr = tensor.detach().cpu()
    if dtype == "float16":
        arr = arr.to(torch.float16)
    elif dtype == "float32":
        arr = arr.to(torch.float32)
    payload = {
        "shape": list(arr.shape),
        "dtype": str(arr.numpy().dtype),
        "data": base64.b64encode(zlib.compress(arr.numpy().tobytes())).decode("ascii"),
    }
    return json.dumps(payload, separators=(",", ":"))


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (torch.log(p.clamp_min(1e-12)) - torch.log(q.clamp_min(1e-12)))).sum(dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def total_variation_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.abs(p - q).sum(dim=-1)


def topk_overlap_and_rank(ref_probs: torch.Tensor, cmp_probs: torch.Tensor, k: int = 10) -> Tuple[int, int]:
    ref_top = torch.topk(ref_probs, k=k, dim=-1).indices.detach().cpu().tolist()
    cmp_top = torch.topk(cmp_probs, k=k, dim=-1).indices.detach().cpu().tolist()
    overlap = len(set(ref_top).intersection(cmp_top))
    ref_top1 = ref_top[0]
    sorted_cmp = torch.argsort(cmp_probs, descending=True)
    rank = int((sorted_cmp == ref_top1).nonzero(as_tuple=False)[0, 0].item()) + 1
    return overlap, rank


def decode_forward_with_hidden(model, inputs_ids):
    hidden_states = model.word_embedding(inputs_ids)
    if model.num_gpus > 1:
        for ldx in range(model.num_layers):
            hidden_states = model.layer_decode(ldx, hidden_states)
            hidden_states = model.parameter_move(hidden_states, ldx)
        hidden_states = hidden_states.to(model.layers[0].device)
    else:
        for ldx in range(model.num_layers):
            hidden_states = model.layer_decode(ldx, hidden_states)
    hidden_states = model.layernorm(hidden_states, model.norm_variance_epsilon, model.norm_weight)
    logits = model.lm(hidden_states)
    return logits, hidden_states


def collect_stage_with_hidden(engine, cache_name, start_token, steps, forced_inputs=None):
    engine._activate_cache(cache_name)
    cur = start_token
    outputs = []
    logits_list = []
    hidden_list = []
    for step_idx in range(steps):
        logits, hidden = decode_forward_with_hidden(engine.model, cur)
        out = engine.model.sampling(logits, do_sample=False)
        outputs.append(out)
        logits_list.append(logits.detach().to(torch.float32))
        hidden_list.append(hidden.detach().to(torch.float32))
        if forced_inputs is not None and step_idx < forced_inputs.shape[1]:
            cur = forced_inputs[:, step_idx:step_idx + 1]
        else:
            cur = out
    return (
        torch.cat(outputs, dim=1),
        torch.cat(logits_list, dim=1),
        torch.cat(hidden_list, dim=1),
    )


def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def append_rows(path: str, fieldnames: List[str], rows: List[Dict[str, object]]):
    ensure_parent_dir(path)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
