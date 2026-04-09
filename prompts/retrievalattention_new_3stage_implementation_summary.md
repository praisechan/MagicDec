# RetrievalAttention_new 3-Stage Hierarchical Self-Spec Implementation Summary

## 1. Purpose of this document
This document explains:
- how the original codebase worked before this implementation,
- how the new 3-stage hierarchical speculative decoding system works now,
- exactly which files and code paths were added or changed.

This is intended as a handoff reference for GPT-Codex to continue development safely.

---

## 2. Baseline architecture before this work

### 2.1 RetrievalAttention_new baseline (non self-spec path)
The original RetrievalAttention_new runtime was primarily a single-model generation pipeline:
- Entry benchmark: Engine/RetrievalAttention_new/benchmark/longbench/pred.py
- Model abstraction: Engine/RetrievalAttention_new/model_hub/LLM.py
- Model implementation: Engine/RetrievalAttention_new/model_hub/llama.py
- KV cache backends:
  - Engine/RetrievalAttention_new/cache_hub/flash_attn_cache.py (full attention)
  - Engine/RetrievalAttention_new/cache_hub/retroinfer_cache.py (retrieval attention)

High-level flow in pred.py:
1. Load dataset and prompt format.
2. Build attention config with generate_config(...).
3. Call llm.generate(...), where attention_type is either Full_Flash_Attn or RetroInfer.
4. generate() allocates one KV cache, runs prefill once, then decode loop.

Important properties of the baseline:
- It was not a hierarchical 3-stage speculate/verify/settle loop.
- It did not maintain separate live caches for draft, early verify, and final verify in one online decoding session.
- It did not include per-step speculative acceptance accounting and stage-level replay/synchronization logic.

### 2.2 Existing self-spec reference style (StreamingLLM)
Reference self-spec behavior existed in tests/StreamingLLM/selfspec_benchmark.py with Draft + Verify mechanics:
- Draft stage autoregressively proposes gamma tokens.
- Verify stage checks them and computes accepted span plus bonus token.
- Cache lengths are manually rolled back/advanced.

This served as interface inspiration, but the RetrievalAttention_new integration required new backend logic because its cache internals and metadata synchronization needs are different.

---

## 3. New implementation overview

The new system introduces:
1. A dedicated RetrievalAttention_new backend API for self-spec:
   - Engine/RetrievalAttention_new/backend.py
2. A dedicated 3-stage benchmark driver:
   - tests/RetrievalAttention_new/selfspec_benchmark.py

Core design:
- Stage 1 Draft: RetroInfer with budget1
- Stage 2 Early Verify: RetroInfer with budget2 (or budget2_high for low-confidence routing)
- Stage 3 Final Verify: Full_Flash_Attn authoritative settlement with a persistent live cache

The loop is hierarchical:
- repeatedly run Draft + Early Verify to grow pending online tokens,
- when pending length reaches gamma2 (or EOS/max-token boundary), run one authoritative Final Verify call,
- settle the chunk and synchronize all stage caches to authoritative state.

---

## 4. How the 3-stage system works now

### 4.1 Stage cache layout
The backend maintains four active caches:
- draft
- early_verify
- early_verify_high
- final_verify

Why four caches:
- draft: fast speculation under budget1
- early_verify: normal verification under budget2
- early_verify_high: stricter verification under budget2_high for low-confidence cases
- final_verify: full attention authoritative reference used online at each settlement boundary

### 4.2 Setup and prefill
In tests/RetrievalAttention_new/selfspec_benchmark.py:
1. preprocess input (pg19 or longbenchv1)
2. engine.setup_caches(...) initializes draft/early_verify/early_verify_high via RetroInfer
3. engine.setup_final_verify_cache(...) initializes final_verify via Full_Flash_Attn
4. engine.encode(...) gets first token from draft prefill state

All stage caches are initialized with the same prefix context but different attention mode/budget configuration.

### 4.3 Iterative online loop
Per iteration:
1. Snapshot cache states for draft and both early-verify caches.
2. Draft stage:
   - speculate tokens using draft cache
   - also compute token confidence margins from logits
3. Dynamic routing decision (if enabled):
   - min_conf > T_high => skip early verify
   - min_conf < T_low => high mode (budget2_high cache)
   - otherwise normal mode (budget2 cache)
4. Early verify behavior:
   - skip mode:
     - do not commit immediately
     - keep early_verify, early_verify_high, and final_verify caches unchanged
     - anchor on the most recent committed token before first skip
     - re-draft from the same anchor with expanding span length gamma1 * n for consecutive skips
       - when skip-buffered span reaches settlement boundary, run one safety early_verify in normal mode first
   - normal/high mode:
     - if there was a skip streak, run one early_verify over the full stacked span from the same anchor
     - apply accepted-prefix + one bonus-token semantics over that full span
5. Revert/replay:
   - skip mode: revert only draft to the skip anchor snapshot for the next re-draft
   - normal/high mode: revert snapshots and replay committed prefix to keep stage caches aligned
   - canonical replay policy: regardless of normal/high routing decision, committed-prefix replay is applied through early_verify (budget2) as the canonical source, then synchronized to draft and early_verify_high
6. Pending span handling:
   - normal/high mode appends proposed online span immediately
   - skip mode appends only when settlement condition is reached
   - safety early_verify does not force immediate final settlement; decoding continues until verified committed accumulation satisfies settlement boundary
   - if the optional rejection indicator is enabled and fires on the current accepted-prefix rows, settlement is forced immediately
7. If settlement boundary reached (gamma2/EOS/max-token):
   - run final_verify once over current pending span,
   - compare pending tokens to final outputs,
   - build authoritative committed span as accepted prefix + final authoritative bonus,
   - revert all three stage families (draft/early/final) to chunk start snapshots,
   - replay authoritative prefix on final cache and synchronize growth into early and draft caches,
   - emit settled tokens and reset pending buffer.

Termination:
- EOS or max token budget.

### 4.4 Bonus-token contract
Early verify and final verify both return proposed span plus one additional next-token prediction.
- On mismatch, accepted span stops before mismatch.
- The token at mismatch position from verifier output becomes authoritative bonus extension.
- Accounting counts verify calls even when only bonus extension is effectively committed.

### 4.5 Dynamic budget routing
Dynamic routing signal:
- per drafted token margin = softmax(top1) - softmax(top2)
- routing uses minimum margin across drafted block

Routing:
- skip: no early verify call, no immediate commit, and buffered anchor-based re-draft growth
- high: verify with early_verify_high cache (budget2_high)
- normal: verify with early_verify cache (budget2)

Replay/sync note:
- high mode affects acceptance decision only (which verifier output is trusted for accepted-prefix + bonus at that step)
- post-verify cache growth replay uses a single canonical source path (early_verify) to reduce cross-cache drift risk

Safety routing note:
- if skip-buffered span hits boundary (gamma2/EOS/max-token), run one normal early_verify on buffered span before any final settlement decision.
- final_verify is still gated by the same settlement condition on committed pending tokens (not just by executing safety early_verify).

Safety note:
- no in-place Python-side nprobe mutation during runtime
- budget switching is cache-based routing, not low-level mutable state mutation

### 4.6 Optional pre-final rejection indicator
`tests/RetrievalAttention_new/selfspec_benchmark.py` now has an optional benchmark-side rejection indicator that can force earlier final settlement before the normal `gamma2` boundary.

CLI surface:
- `--rejection_indicator {disabled,margin_kl_threshold,margin_only}`
- `--ri_margin_threshold`
- `--ri_accepted_mod_kl_threshold`
- `--ri_accepted_drift_count`

How it is evaluated:
- the indicator is only evaluated on accepted-prefix rows from the early-verify stage
- it never uses final-verify outputs as an input signal
- state is accumulated over the current unsettled block and reset after each final settlement

Modes:
- `disabled`: no indicator logic
- `margin_kl_threshold`: count accepted-prefix rows where `early_margin < ri_margin_threshold` and `KL(early_verify || draft) >= ri_accepted_mod_kl_threshold`
- `margin_only`: count accepted-prefix rows where `early_margin < ri_margin_threshold`

Trigger rule:
- force settlement when the counted accepted-prefix rows reach `ri_accepted_drift_count`

Implementation note:
- there is no separate action flag anymore; if the indicator is enabled and triggers, the runtime always takes the same action: immediate final settlement on the current pending span

---

## 5. Cache consistency and synchronization mechanics

Implemented in Engine/RetrievalAttention_new/backend.py.

### 5.1 Snapshot/revert/truncate
Backend provides:
- snapshot_state(cache_name)
- revert_to(cache_name, snapshot)
- truncate_to(cache_name, target_context)

Snapshot captures:
- context length
- static_pattern_total (when present)
- valid_length_dict per device (when present)

### 5.2 Commit and replay
commit_prefix(...) supports:
- direct replay in target cache
- replay in source cache plus synchronized growth copy into target cache
- source/target synchronization for stage alignment after settlement

### 5.3 Cross-cache growth sync
_sync_committed_growth(...) synchronizes:
- context
- steady-zone KV region (when available)
- valid lengths
- RetroInfer retrieval metadata tensors when compatible:
  - centroids
  - value_sum
  - centroids_mask
  - cluster_size

This reduces divergence risk between stage caches after authoritative settlement.

### 5.4 Multi-GPU lifecycle hardening
Added CUDA synchronization boundaries in backend cache lifecycle:
- before each stage cache init
- after prefill forward
- after model.move()
- after RetroInfer graph capture
- before clear_kv teardown

Goal:
- avoid async overlap hazards during multi-cache setup/teardown in multi-GPU mode.

---

## 6. What changed, file by file

## 6.1 Added: Engine/RetrievalAttention_new/backend.py
New LMBackend API for RetrievalAttention_new self-spec integration.

Major methods:
- load_model
- preprocess_input
- setup_caches
- setup_final_verify_cache
- encode
- speculate / speculate_with_confidence
- verify / early_verify / final_verify
- snapshot_state / revert_to / truncate_to
- commit_prefix / cache_state_report
- clear_kv / delete_cache

Key implementation points:
- multi-cache stage orchestration
- dynamic confidence-based stage-2 routing
- cache replay and synchronization helpers
- retrieval metadata synchronization
- multi-GPU synchronization barriers

## 6.2 Added: tests/RetrievalAttention_new/selfspec_benchmark.py
Dedicated benchmark runner for 3-stage hierarchical self-spec.

Major behavior:
- dataset/model config loading for RetrievalAttention_new
- per-step online Draft -> Early Verify -> Final Verify settlement
- dynamic routing with skip/high/normal stage-2 decisions
- optional accepted-prefix rejection-indicator forcing early settlement
- helper-based refactor for repeated verify/commit/replay/skip-reset flows
- canonical replay-source policy after verification to keep draft/normal/high verifier caches consistent when high mode is selected
- CSV accumulation logging (stage output JSON logging removed)
- per-step and final console statistics

Outputs:
- step_log.csv
- accumulated_log.csv

## 6.3 Existing core internals reused (not replaced)
The new backend relies on existing RetrievalAttention_new internals:
- Engine/RetrievalAttention_new/model_hub/LLM.py
- Engine/RetrievalAttention_new/model_hub/llama.py
- Engine/RetrievalAttention_new/cache_hub/retroinfer_cache.py
- Engine/RetrievalAttention_new/cache_hub/flash_attn_cache.py

No old RetrievalAttention path is used for this implementation path.

---

## 7. Control-flow comparison: original vs new

Original RetrievalAttention_new benchmark path:
- one attention mode per run
- one KV cache lifecycle per generate call
- no speculative hierarchy

New self-spec path:
- concurrent stage-specific cache families
- hierarchical online decoding with gamma1 drafting and gamma2 settlement
- explicit cache snapshot/revert/replay/sync operations
- dynamic early-verify budget routing
- authoritative final settlement with live full-attention cache

---

## 8. Metrics and logging map

In tests/RetrievalAttention_new/selfspec_benchmark.py:
- per-iteration stage timing and accepted/rejected counts are printed
- dynamic routing counters tracked (skip/high/normal)
- min confidence tracked when dynamic mode enabled
- speculative/verify/final call counts accumulated
- generated token counts accumulated
- step and accumulated CSV rows written
- rejection-indicator configuration is logged in CSV when present

Practical meaning of call counters:
- speculate_calls counts effective draft decode span per decision point
   - for consecutive skip re-drafts from the same anchor, counting is replacement-style (latest gamma1 * n), not cumulative sum across rewinds
- early_verify_calls counts actual early verify invocations (skip does not increment)
- final_verify_calls counts chunk-level settlement events

Current rejection-indicator CSV fields:
- `rejection_indicator`
- `ri_margin_threshold`
- `ri_accepted_mod_kl_threshold`
- `ri_accepted_drift_count`

---

## 9. Key invariants for future modifications

1. Stage cache coherence
- After any authoritative settlement, draft/early/final caches must represent the same committed prefix state.

2. Bonus token semantics
- Verify stages are compared on prefix and contribute one bonus token; mismatch handling must preserve this rule.

3. Settlement granularity
- final_verify_calls must correspond to settlement boundaries (gamma2/EOS/max-token), not every early verify iteration.

4. Skip buffering semantics
- skip mode must not mutate verifier caches through immediate commit/replay.
- consecutive skip decisions must preserve anchor-token continuity and expanding re-draft span behavior.

5. Dynamic routing safety
- Keep budget switching as explicit cache routing, not ad-hoc mutable low-level cache parameter mutation.
- Keep post-verify replay on a canonical source path so high-mode routing does not introduce separate replay semantics.

6. Rejection-indicator scope
- Keep rejection-indicator features limited to pre-final signals available before authoritative settlement.
- Keep indicator state chunk-local so it resets after each final settlement.

7. Multi-GPU ordering
- Preserve CUDA synchronization boundaries around cache init/move/capture/teardown operations.

---

## 10. Suggested reading order for next GPT-Codex step

1. Engine/RetrievalAttention_new/backend.py
2. tests/RetrievalAttention_new/selfspec_benchmark.py
3. Engine/RetrievalAttention_new/model_hub/llama.py
4. Engine/RetrievalAttention_new/model_hub/LLM.py
5. Engine/RetrievalAttention_new/cache_hub/retroinfer_cache.py
6. Engine/RetrievalAttention_new/cache_hub/flash_attn_cache.py
7. Engine/RetrievalAttention_new/benchmark/longbench/pred.py

This order gives the fastest path from high-level orchestration to low-level cache/attention behavior.

---

## 11. Current version notes (latest)

1. High-budget replay consistency update
- In tests/RetrievalAttention_new/selfspec_benchmark.py, replay_verified_prefix now uses early_verify as canonical replay source for committed-prefix growth synchronization, including when dynamic mode chooses high verification.

2. Why this was changed
- High mode should increase or preserve verification quality, not worsen final settlement consistency due to divergent replay-source behavior.
- Canonical replay semantics reduce the chance that routing choice itself changes cache synchronization behavior.

3. Debugging caveat for reproduction
- If runtime logs show Initial n_centroids: 0, RetroInfer retrieval-index behavior may not be active in that run configuration, and normal/high differences can appear minimal.
- To evaluate budget2 vs budget2_high impact, use settings where retrieval indexing is active.

4. Rejection-indicator implementation update
- The benchmark now supports two implementation modes for pre-final forcing:
  - `margin_kl_threshold`
  - `margin_only`
- The older action flag was removed from the implementation path; enabling the indicator now always means force settlement on trigger.
- The documented CSV surface was reduced to the currently used indicator fields only.