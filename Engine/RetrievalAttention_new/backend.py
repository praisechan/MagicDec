import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import torch

RA_NEW_ROOT = os.path.abspath(os.path.dirname(__file__))
if RA_NEW_ROOT not in sys.path:
    sys.path.append(RA_NEW_ROOT)

from MagicDec.Engine.RetrievalAttention_new.config import generate_config
from MagicDec.Engine.RetrievalAttention_new.model_hub import load_model


@dataclass
class CacheSnapshot:
    context: int
    static_pattern_total: Optional[int] = None
    valid_length_dict: Optional[Dict[str, torch.Tensor]] = None


class LMBackend:
    def __init__(self, dtype=torch.bfloat16, device="cuda", dec_len=8):
        self.dtype = dtype
        self.device = device
        self.runtime_device = self._normalize_runtime_device(device)
        self.model_device = None
        self.dec_len = dec_len

        self.model = None
        self.model_path = None
        self.max_len = None

        self.input_ids = None
        self.attention_masks = None
        self.valid_start = None

        self.caches = {}
        self.cache_meta = {}
        self.prefill_tokens = {}
        self._skip_next_early_commit = False

    def _softmax_top2_margin(self, logits):
        probs = torch.softmax(logits.float(), dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        return top2[..., 0] - top2[..., 1]

    def _ensure_feature_axis(self, tensor):
        if tensor.dim() == 1:
            return tensor.unsqueeze(1)
        return tensor

    def _confidence_feature_dict(self, logits):
        probs = torch.softmax(logits.float(), dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        top1_prob = self._ensure_feature_axis(top2[..., 0])
        top2_prob = self._ensure_feature_axis(top2[..., 1])
        entropy = self._ensure_feature_axis(
            -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        )
        return {
            "top1_prob": top1_prob,
            "top2_prob": top2_prob,
            "margin": top1_prob - top2_prob,
            "entropy": entropy,
        }

    def _synchronize_cuda(self):
        if not torch.cuda.is_available():
            return
        device_count = torch.cuda.device_count()
        for didx in range(device_count):
            torch.cuda.synchronize(didx)

    def _normalize_runtime_device(self, device):
        if device in (None, "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            return "cuda"
        return device

    def _normalize_model_device(self, device):
        if device == "cuda":
            if torch.cuda.device_count() > 1:
                return "auto"
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device

    def _resolve_runtime_device(self):
        if self.model is not None and getattr(self.model, "layers", None):
            return self.model.layers[0].device
        return self._normalize_runtime_device(self.device)

    def load_model(self, model_path, max_len, dtype, device, bsz):
        self.model_device = self._normalize_model_device(device)
        self.model = load_model(model_path, max_len, dtype=dtype, device=self.model_device)
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        self.model.tokenizer.padding_side = "left"
        self.model_path = model_path
        self.max_len = max_len
        self.batch_size = bsz
        self.runtime_device = self._resolve_runtime_device()

    def preprocess_input(
        self,
        data,
        prompt_format,
        dataset="pg19",
        seq_len=None,
    ):
        if dataset not in {"longbenchv1", "pg19"}:
            raise ValueError(f"Unsupported dataset: {dataset}")

        if dataset == "pg19" and "{text}" in prompt_format:
            prefix, _, suffix = prompt_format.partition("{text}")
            raw_text = data.get("text", "")

            # Keep a single special-token prefix, then append raw spans without extra specials.
            prefix_ids = self.model.tokenizer.encode(prefix, return_tensors="pt", add_special_tokens=True)
            text_ids = self.model.tokenizer.encode(raw_text, return_tensors="pt", add_special_tokens=False)
            suffix_ids = self.model.tokenizer.encode(suffix, return_tensors="pt", add_special_tokens=False)

            if seq_len is not None:
                reserve_for_suffix = suffix_ids.shape[1]
                max_prefix_text = max(seq_len - reserve_for_suffix, 0)
                prefix_text_ids = torch.cat([prefix_ids, text_ids], dim=1)
                prefix_text_ids = prefix_text_ids[:, :max_prefix_text]
                input_ids = torch.cat([prefix_text_ids, suffix_ids], dim=1)
                if input_ids.shape[1] > seq_len:
                    input_ids = input_ids[:, -seq_len:]
            else:
                input_ids = torch.cat([prefix_ids, text_ids, suffix_ids], dim=1)
        else:
            prompt = prompt_format.format(**data)
            inputs = self.model.tokenizer([prompt], return_tensors="pt", padding=True)
            input_ids = inputs.input_ids

        self.attention_masks = torch.ones_like(input_ids, device=input_ids.device).to(self.runtime_device)
        input_ids = input_ids.to(self.runtime_device)

        return input_ids

    def _build_attn_config(self, seq_len, attention_type, retrieval_budget, estimation_budget):
        if seq_len > 16100 and seq_len < 16500:
            gpu_only = False # Monkey patch to avoid crash due to BUILD_SEGMENT threshold error related to static pattern total..
        else:
            gpu_only = True
        return generate_config(
            self.model_path,
            seq_len,
            attention_type,
            retrieval_budget=retrieval_budget,
            estimation_budget=estimation_budget,
            gpu_only=gpu_only
        )

    def _init_single_cache(self, cache_name, attention_type, retrieval_budget, estimation_budget, max_new_tokens):
        seq_len = self.input_ids.shape[1]
        bsz = self.input_ids.shape[0]

        # In multi-GPU mode, flush async work between stage-cache initializations.
        self._synchronize_cuda()

        self.model.attention_type = attention_type
        self.model.batch_size = bsz
        self.model.input_length = seq_len
        self.model.max_new_length = max_new_tokens
        self.model.prefill_bsz = min(1, bsz)
        self.model.prefill_method = "full"

        attn_config = self._build_attn_config(
            seq_len,
            attention_type,
            retrieval_budget,
            estimation_budget,
        )
        self.model.init_kv_cache(self.valid_start, attn_config)

        logits = self.model.prefill_forward(self.input_ids)
        first_token = self.model.sampling(logits, do_sample=False)
        self._synchronize_cuda()
        self.model.move()
        self._synchronize_cuda()

        if attention_type == "RetroInfer":
            self.model.kv_cache.capture_cuda_graph()
            self._synchronize_cuda()

        self.caches[cache_name] = self.model.kv_cache
        self.cache_meta[cache_name] = {
            "attention_type": attention_type,
            "retrieval_budget": retrieval_budget,
            "estimation_budget": estimation_budget,
            "max_new_tokens": max_new_tokens,
        }
        self.prefill_tokens[cache_name] = first_token

    def setup_caches(
        self,
        input_ids,
        attention_masks,
        budget1,
        budget2,
        estimate_ratio,
        max_new_tokens,
        budget2_high=None,
    ):
        self.input_ids = input_ids.to(self.runtime_device)
        self.attention_masks = attention_masks.to(self.runtime_device)
        self.valid_start = (
            self.attention_masks.shape[1] - torch.sum(self.attention_masks, dim=-1).detach().cpu().numpy()
        )

        self._init_single_cache(
            cache_name="draft",
            attention_type="RetroInfer",
            retrieval_budget=budget1,
            estimation_budget=estimate_ratio,
            max_new_tokens=max_new_tokens,
        )
        self._init_single_cache(
            cache_name="early_verify",
            attention_type="RetroInfer",
            retrieval_budget=budget2,
            estimation_budget=estimate_ratio,
            max_new_tokens=max_new_tokens,
        )
        self._init_single_cache(
            cache_name="early_verify_high",
            attention_type="RetroInfer",
            retrieval_budget=budget2 if budget2_high is None else budget2_high,
            estimation_budget=estimate_ratio,
            max_new_tokens=max_new_tokens,
        )

    def setup_final_verify_cache(self, input_ids, attention_masks, max_new_tokens):
        self.input_ids = input_ids.to(self.runtime_device)
        self.attention_masks = attention_masks.to(self.runtime_device)
        self.valid_start = (
            self.attention_masks.shape[1] - torch.sum(self.attention_masks, dim=-1).detach().cpu().numpy()
        )

        self._init_single_cache(
            cache_name="final_verify",
            attention_type="Full_Flash_Attn",
            retrieval_budget=1.0,
            estimation_budget=0.0,
            max_new_tokens=max_new_tokens,
        )

    def prepare_prefill_state(self, input_ids, attention_masks):
        self.input_ids = input_ids.to(self.runtime_device)
        self.attention_masks = attention_masks.to(self.runtime_device)
        self.valid_start = (
            self.attention_masks.shape[1] - torch.sum(self.attention_masks, dim=-1).detach().cpu().numpy()
        )

    def _activate_cache(self, cache_name):
        meta = self.cache_meta[cache_name]
        self.model.attention_type = meta["attention_type"]
        self.model.kv_cache = self.caches[cache_name]

    def _replay_tokens(self, cache_name, replay):
        self._activate_cache(cache_name)
        for idx in range(replay.shape[1]):
            _ = self.model.decode_forward(replay[:, idx:idx + 1])

    def _copy_tensor_list_growth(self, source_list, target_list, old_len, new_len):
        if source_list is None or target_list is None:
            return
        layer_count = min(len(source_list), len(target_list))
        for ldx in range(layer_count):
            src = source_list[ldx]
            tgt = target_list[ldx]
            if src is None or tgt is None:
                continue

            axis = 1 if src.dim() >= 3 else -1
            src_len = src.shape[axis]
            tgt_len = tgt.shape[axis]
            if old_len >= min(src_len, tgt_len):
                continue
            copy_end = min(new_len, src_len, tgt_len)
            if copy_end <= old_len:
                continue

            if axis == 1:
                tgt[:, old_len:copy_end, ...].copy_(src[:, old_len:copy_end, ...])
            else:
                tgt[..., old_len:copy_end].copy_(src[..., old_len:copy_end])

    def _sync_committed_growth(self, source_cache_name, target_cache_name, old_target_context):
        source = self.caches[source_cache_name]
        target = self.caches[target_cache_name]

        new_context = int(source.context)
        target.context = new_context

        if hasattr(source, "static_pattern_total") and hasattr(target, "static_pattern_total"):
            old_static = int(getattr(target, "static_pattern_total"))
            new_static = int(getattr(source, "static_pattern_total"))

            if (
                hasattr(source, "steady_zone_keys")
                and hasattr(target, "steady_zone_keys")
                and hasattr(source, "steady_zone_values")
                and hasattr(target, "steady_zone_values")
                and new_static >= old_static
            ):
                layer_count = min(len(source.steady_zone_keys), len(target.steady_zone_keys))
                for ldx in range(layer_count):
                    target.steady_zone_keys[ldx][:, :, old_static:new_static, :].copy_(
                        source.steady_zone_keys[ldx][:, :, old_static:new_static, :]
                    )
                    target.steady_zone_values[ldx][:, :, old_static:new_static, :].copy_(
                        source.steady_zone_values[ldx][:, :, old_static:new_static, :]
                    )

            target.static_pattern_total = new_static

        # Copy RetroInfer index structures so target retrieval uses equivalent metadata.
        if hasattr(source, "n_centroids") and hasattr(target, "n_centroids"):
            src_centroids = int(source.n_centroids)
            tgt_centroids = int(target.n_centroids)

            if src_centroids == tgt_centroids:
                self._copy_tensor_list_growth(
                    getattr(source, "centroids", None),
                    getattr(target, "centroids", None),
                    0,
                    src_centroids,
                )
                self._copy_tensor_list_growth(
                    getattr(source, "value_sum", None),
                    getattr(target, "value_sum", None),
                    0,
                    src_centroids,
                )
                self._copy_tensor_list_growth(
                    getattr(source, "centroids_mask", None),
                    getattr(target, "centroids_mask", None),
                    0,
                    src_centroids,
                )
                self._copy_tensor_list_growth(
                    getattr(source, "cluster_size", None),
                    getattr(target, "cluster_size", None),
                    0,
                    src_centroids,
                )
            else:
                # If index width diverges (e.g., update boundary), avoid stale skip behavior.
                self._skip_next_early_commit = False

        if hasattr(source, "valid_length_dict") and hasattr(target, "valid_length_dict"):
            for dev, tensor in source.valid_length_dict.items():
                if dev in target.valid_length_dict:
                    target.valid_length_dict[dev].copy_(tensor)
            target.valid_length = target.valid_length_dict[target.layer_mapping[str(0)]]

        if new_context < old_target_context:
            raise RuntimeError("Committed replay sync produced a shorter target context")

    def encode(self, input_ids=None):
        if input_ids is not None:
            self.input_ids = input_ids.to(self.runtime_device)
        return self.prefill_tokens["draft"]

    def _decode_steps(
        self,
        cache_name,
        start_token,
        steps,
        forced_inputs=None,
        return_confidence=False,
        return_features=False,
    ):
        self._activate_cache(cache_name)
        cur = start_token
        outputs = []
        margins = []
        feature_lists = None

        if return_features:
            feature_lists = {
                "top1_prob": [],
                "top2_prob": [],
                "margin": [],
                "entropy": [],
            }

        for step_idx in range(steps):
            logits = self.model.decode_forward(cur)
            out = self.model.sampling(logits, do_sample=False)
            outputs.append(out)
            if return_confidence:
                margins.append(self._softmax_top2_margin(logits))
            if return_features:
                features = self._confidence_feature_dict(logits)
                for name, value in features.items():
                    feature_lists[name].append(value)

            if forced_inputs is not None and step_idx < forced_inputs.shape[1]:
                cur = forced_inputs[:, step_idx:step_idx + 1]
            else:
                cur = out

        if outputs:
            outputs = torch.cat(outputs, dim=1)
        else:
            outputs = torch.empty((start_token.shape[0], 0), dtype=start_token.dtype, device=start_token.device)

        if return_confidence:
            if margins:
                confidence = torch.cat(margins, dim=1)
                min_conf = torch.min(confidence).item()
            else:
                confidence = torch.empty((start_token.shape[0], 0), dtype=torch.float32, device=start_token.device)
                min_conf = float("inf")
            return outputs, confidence, min_conf

        if return_features:
            feature_dict = {}
            for name, values in feature_lists.items():
                if values:
                    feature_dict[name] = torch.cat(values, dim=1)
                else:
                    feature_dict[name] = torch.empty(
                        (start_token.shape[0], 0),
                        dtype=torch.float32,
                        device=start_token.device,
                    )
            return outputs, feature_dict

        return outputs

    def speculate(self, current_token, gamma):
        return self._decode_steps("draft", current_token, gamma)

    def speculate_with_confidence(self, current_token, gamma):
        return self._decode_steps("draft", current_token, gamma, return_confidence=True)

    def speculate_with_features(self, current_token, gamma):
        return self._decode_steps("draft", current_token, gamma, return_features=True)

    def verify(self, current_token, draft_tokens):
        return self.early_verify(current_token, draft_tokens)

    def early_verify(self, current_token, draft_tokens, mode="normal"):
        if mode == "high":
            cache_name = "early_verify_high"
        else:
            cache_name = "early_verify"
        return self._decode_steps(
            cache_name=cache_name,
            start_token=current_token,
            steps=draft_tokens.shape[1] + 1,
            forced_inputs=draft_tokens,
        )

    def final_verify(self, current_token, proposed_tokens):
        return self._decode_steps(
            cache_name="final_verify",
            start_token=current_token,
            steps=proposed_tokens.shape[1] + 1,
            forced_inputs=proposed_tokens,
        )

    def snapshot_state(self, cache_name):
        cache = self.caches[cache_name]
        snapshot = CacheSnapshot(context=int(cache.context))

        if hasattr(cache, "static_pattern_total"):
            snapshot.static_pattern_total = int(cache.static_pattern_total)

        if hasattr(cache, "valid_length_dict"):
            snapshot.valid_length_dict = {
                dev: tensor.clone() for dev, tensor in cache.valid_length_dict.items()
            }

        return snapshot

    def revert_to(self, cache_name, snapshot):
        cache = self.caches[cache_name]
        cache.context = int(snapshot.context)

        if snapshot.static_pattern_total is not None and hasattr(cache, "static_pattern_total"):
            cache.static_pattern_total = int(snapshot.static_pattern_total)

        if snapshot.valid_length_dict is not None and hasattr(cache, "valid_length_dict"):
            for dev, tensor in snapshot.valid_length_dict.items():
                cache.valid_length_dict[dev].copy_(tensor)
            cache.valid_length = cache.valid_length_dict[cache.layer_mapping[str(0)]]

    def truncate_to(self, cache_name, target_context):
        cache = self.caches[cache_name]
        target_context = int(target_context)
        if target_context > int(cache.context):
            raise ValueError("truncate_to only supports truncating to a smaller context")

        delta = int(cache.context) - target_context
        cache.context = target_context

        if hasattr(cache, "static_pattern_total"):
            cache.static_pattern_total = max(0, int(cache.static_pattern_total) - delta)

        if hasattr(cache, "valid_length_dict"):
            for dev in cache.valid_length_dict:
                cache.valid_length_dict[dev].sub_(delta)
                cache.valid_length_dict[dev].clamp_(min=0)
            cache.valid_length = cache.valid_length_dict[cache.layer_mapping[str(0)]]

    def commit_prefix(
        self,
        cache_name,
        current_token,
        accepted_prefix,
        sync_source=None,
        replay_source=None,
        prefer_draft_replay=False,
        source_replayed=False,
    ):
        if cache_name == "early_verify" and self._skip_next_early_commit:
            self._skip_next_early_commit = False
            return

        replay = current_token if accepted_prefix.numel() == 0 else torch.cat([current_token, accepted_prefix], dim=1)

        if sync_source is not None:
            if sync_source not in self.caches:
                raise KeyError(f"Unknown sync source cache: {sync_source}")

            if (
                sync_source != cache_name
                and hasattr(self.caches[sync_source], "steady_zone_keys")
                and hasattr(self.caches[cache_name], "steady_zone_keys")
            ):
                old_target_context = int(self.caches[cache_name].context)
                if not source_replayed:
                    self._replay_tokens(sync_source, replay)
                self._sync_committed_growth(sync_source, cache_name, old_target_context)
                return

            self._replay_tokens(cache_name, replay)
            return

        # For draft commit replay, use early_verify budget path then synchronize growth to draft.
        if (
            not prefer_draft_replay
            and
            cache_name == "draft"
            and (replay_source in self.caches if replay_source is not None else "early_verify" in self.caches)
            and self.cache_meta.get("draft", {}).get("attention_type") == "RetroInfer"
            and self.cache_meta.get(replay_source or "early_verify", {}).get("attention_type") == "RetroInfer"
        ):
            source_name = replay_source or "early_verify"
            old_draft_context = int(self.caches["draft"].context)
            self._replay_tokens(source_name, replay)
            self._sync_committed_growth(source_name, "draft", old_draft_context)
            # Legacy flow calls commit_prefix("early_verify", ...) immediately after draft commit.
            self._skip_next_early_commit = replay_source is None and source_name == "early_verify"
            return

        self._replay_tokens(cache_name, replay)

    def cache_state_report(self, cache_name):
        cache = self.caches[cache_name]
        report = {
            "cache_name": cache_name,
            "context": int(cache.context),
        }
        if hasattr(cache, "static_pattern_total"):
            report["static_pattern_total"] = int(cache.static_pattern_total)
        if hasattr(cache, "valid_length_dict"):
            report["valid_lengths"] = {
                dev: tensor.detach().cpu().tolist() for dev, tensor in cache.valid_length_dict.items()
            }
        return report

    def clear_kv(self):
        self._synchronize_cuda()
        self.caches = {}
        self.cache_meta = {}
        self.prefill_tokens = {}
        self._skip_next_early_commit = False
        if self.model is not None:
            self.model.kv_cache = None
        torch.cuda.empty_cache()

    def delete_cache(self, cache_name):
        if cache_name in self.caches:
            del self.caches[cache_name]
        if cache_name in self.cache_meta:
            del self.cache_meta[cache_name]
        if cache_name in self.prefill_tokens:
            del self.prefill_tokens[cache_name]

        if self.model is not None and getattr(self.model, "kv_cache", None) is not None:
            if cache_name == "final_verify":
                self.model.kv_cache = None
        torch.cuda.empty_cache()
