# Phase 2 Summary

## Conclusions reached

Phase 2 confirms that final-verify rejection is not just a shallow confidence-threshold problem. The clearest signal in the real probes was hidden-state drift: early-verify and full-attention representations diverged much more in rejected blocks than in accepted blocks. In the hidden-state probe, accepted positions had cosine similarity 0.639 on average and L2 drift 91.3, while rejected-block positions dropped to cosine 0.325 and L2 drift 163.0. The directly observed rejected position had cosine 0.190 and L2 drift 179.45, and the immediately preceding position was even more distorted.

Logit/distribution analysis supports the same mechanism, but with a more heterogeneous surface pattern. When same-cycle early logits were available, rejected-block positions almost always had zero top-10 overlap with final verify and poor rank of final verify's top token in the early distribution. However, accepted positions were bimodal: some matched final verify almost exactly, while others still had severe distribution mismatch despite eventual acceptance. That means distributional fragility is real, but not yet a clean standalone indicator.

Budget sensitivity around budget2 was weaker than expected. Replaying the same span at budgets 0.08, 0.10, and 0.12 did not change the early argmax at any observed rejected position; all stayed wrong versus final verify. This suggests the dominant failure mode in these sampled rejections is a stable retrieval miss or approximation distortion, not a marginal case recoverable by a modest budget bump.

## Data files and contents

- `tests/RetrievalAttention_new/research/data/phase2b_logit_probe_Meta-Llama-3.1-8B_32768.csv`
  Per-position draft/early/final logits, softmax payloads, divergence metrics, and rank/overlap features for 5 settled cycles (137 rows, 2 rejected cycles).
- `tests/RetrievalAttention_new/research/data/phase2c_hidden_state_probe_Meta-Llama-3.1-8B_32768.csv`
  Per-position early-vs-final hidden-state cosine similarity and norm drift for 6 cycles (42 rows, 4 rejected cycles).
- `tests/RetrievalAttention_new/research/data/phase2d_budget_sensitivity_Meta-Llama-3.1-8B_32768.csv`
  Per-position low/mid/high budget argmax, margin, and entropy sensitivity for 6 cycles (157 rows, 3 rejected cycles).

## Gating decision for Phase 3

Phase 2 is complete and the next gate is clear: Phase 3 should scale up data collection with a profiler that preserves per-position alignment between pre-final features and the final settled span. The main lesson to carry forward is that we need better logging around the actual mismatch positions themselves. Phase 2b showed that final settlement can outrun the last same-cycle early/draft window, so the Phase 3 profiler must explicitly attach pre-final draft and early-verify features to every settled position, especially the final mismatch position and its neighbors.
