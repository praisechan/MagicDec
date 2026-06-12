# Qwen3-30B-A3B Support Implementation Report

**Date:** 2026-06-11  
**Branch:** `optimize`  
**Status:** Verified working end-to-end

---

## Objective

Add support for the Qwen3-30B-A3B Mixture-of-Experts model to the MagicDec RetrievalAttention_new pipeline, which previously only supported Qwen2.5 (dense) and Llama 3.1 models.

---

## Architectural Differences: Qwen3-30B-A3B vs Qwen2.5

| Feature | Qwen2.5 | Qwen3-30B-A3B |
|---|---|---|
| MLP type | Dense (gate + up + down) | MoE: 128 experts, top-8 routing per token |
| QKV bias | Yes | No |
| QK normalization | None | Per-head RMSNorm after Q/K projection |
| head_dim | `hidden_size / num_heads` | Explicit (128); Q proj expands: 2048 -> 4096 |
| hidden_size | 3584-8192 | 2048 |
| num_hidden_layers | 28-80 | 48 |
| RoPE scaling | YaRN for context > 32K | Standard (no scaling), max 40960 |
| HF model class | `Qwen2ForCausalLM` | `Qwen3MoeForCausalLM` |
| Total / active params | 7B-72B dense | 30B total / ~3B active |

---

## Files Changed

### New Files

1. **`Engine/RetrievalAttention_new/model_hub/qwen3_moe.py`**  
   Core model implementation containing:
   - `Qwen3MoeLayer`: Extracts weights from HuggingFace model per layer:
     - Concatenated QKV weights (no bias)
     - QK RMSNorm weights (`q_norm_weight`, `k_norm_weight`)
     - MoE router gate weight
     - Stacked expert weights: `expert_gate_up_proj` [128, 1536, 2048] and `expert_down_proj` [128, 2048, 768]
     - Layer norm weights
   - `Qwen3MoeModel(LLM)`: Full model class with key method overrides:
     - `wqkv()`: No bias, correct split sizes for expanded Q dim (4096 vs 512 for K/V), per-head QK RMSNorm
     - `wo()`: Reshapes using `num_heads * head_dim` (4096) instead of `hidden_size` (2048) before output projection
     - `mlp()`: MoE routing — softmax gating, top-8 expert selection, normalized weights, expert-parallel dispatch with `silu_and_mul` activation
     - `_set_cos_sin_cache()`: Standard RoPE without YaRN
     - All other methods (attention, KV cache, layernorm, RoPE, parameter_move) follow the same patterns as existing models

2. **`Engine/RetrievalAttention_new/config/Qwen3-30B-A3B.json`**  
   RetroInfer configuration (based on Qwen2.5-32B template).

3. **`tests/RetrievalAttention_new/scripts/0411/qmsum_kl_qwen3_30b_a3b.sh`**  
   Verification benchmark script adapted from `qmsum_kl_32b.sh`.

### Modified Files

4. **`Engine/RetrievalAttention_new/model_hub/__init__.py`**  
   - Added `Qwen3MoeModel` import
   - Added dispatch: `'Qwen3' in model_name and 'A3B' in model_name` (checked before generic `'Qwen'`)
   - Added `Qwen/Qwen3-30B-A3B` to the `add_model_args` choices list

5. **`Engine/RetrievalAttention_new/benchmark/longbench/config/model2path.json`**  
   Added entry: `"qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B"`

6. **`Engine/RetrievalAttention_new/benchmark/longbench/config/model2maxlen.json`**  
   Added entry: `"qwen3-30b-a3b": 40960`

---

## Design Decisions

1. **New file instead of modifying `qwen.py`**: The MoE layer, QK RMSNorm, expanded head_dim, and no-bias Q/K/V are fundamental enough changes that a separate `qwen3_moe.py` is cleaner and avoids risk to existing Qwen2.5 support.

2. **Expert weight stacking**: All 128 expert weights are stacked into tensors `[num_experts, ...]` per layer for efficient indexed access during expert-parallel dispatch. Each expert's gate + up projections are concatenated before stacking.

3. **Expert-parallel MoE dispatch**: Tokens are routed to experts, then for each active expert, assigned tokens are gathered, processed through gate_up -> silu_and_mul -> down, and scattered back with weighted accumulation. This is efficient for small batch sizes (decode: 1 token, 8 experts).

4. **`reshape()` over `view()`**: After `qkv.split()`, the resulting tensors may not be contiguous. Using `.reshape()` handles both contiguous and non-contiguous cases without needing explicit `.contiguous()` calls.

5. **max_length = 40960**: Uses the model's native `max_position_embeddings` without YaRN. YaRN support can be added later if longer contexts are needed.

---

## Verification Results

Ran `qmsum_kl_qwen3_30b_a3b.sh` on the LongBench qmsum task with:
- `prefix_len=8192`, `gamma1=6`, `gamma2=32`
- `budget1=0.02`, `budget2=0.20`
- Rejection indicator: `margin_kl_threshold`

**Results from first 2 completed steps:**

| Metric | Step 0 | Step 1 |
|---|---|---|
| Tokens generated | 100 | 100 |
| Speculate calls | 84 | 84 |
| Early verify calls | 14 | 14 |
| Final verify calls | 4 | 5 |
| Early verify acceptance | 100% | 100% |
| Rejection indicator triggers | 1 | 3 |
| Time per step | ~3.6 min | ~3.6 min |

The model produces coherent text output. All three stages of the speculative decoding pipeline (draft, early verify, final verify) work correctly. The MoE routing, QK RMSNorm, expanded head_dim attention, and RetroInfer KV cache all operate without errors.

---

## How to Run

```bash
# Single KL threshold test
CUDA_VISIBLE_DEVICES=0 python tests/RetrievalAttention_new/selfspec_benchmark_wo_high.py \
  --dataset longbenchv1 \
  --task qmsum \
  --model_name qwen3-30b-a3b \
  --prefix_len 8192 \
  --num_max_token 100 \
  --gamma1 6 --gamma2 32 \
  --budget1 0.02 --budget2 0.20 \
  --estimate_ratio 0.25 \
  --rejection_indicator margin_kl_threshold \
  --ri_margin_threshold 0.01 \
  --ri_accepted_mod_kl_threshold 0.01

# Full sweep
bash tests/RetrievalAttention_new/scripts/0411/qmsum_kl_qwen3_30b_a3b.sh
```

---

## Future Work

- Profile and add model-specific xattn thresholds and minfer patterns for Qwen3-30B-A3B (currently using defaults)
- Add YaRN RoPE scaling support for contexts beyond 40960 tokens
- Consider optimized MoE kernels (e.g., grouped GEMM) for faster expert dispatch during prefill
- Add support for other Qwen3 MoE variants if released
