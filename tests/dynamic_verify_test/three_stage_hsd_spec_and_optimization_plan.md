# 3-Stage Hierarchical Self-Speculative Decoding: Current Behavior and Optimization Specification

## Scope
This document specifies:
1. How the current 3-stage pipeline works in detail.
2. Where redundant prefill occurs and why latency is high.
3. A compatible optimization design that removes redundant prefill by reusing KV cache state.
4. Required verification criteria to prove equivalence with the existing implementation.

Primary runtime entrypoints:
- `tests/dynamic_verify_test/run_3step_profile_optimized.py`
- `Engine/RetrievalAttention/backend_for_3stage.py`
- `Engine/RetrievalAttention/model_hub/LLM.py`
- `Engine/RetrievalAttention/model_hub/llama.py`
- `Engine/RetrievalAttention/model_hub/qwen.py`
- `Engine/RetrievalAttention/cache_hub/retroinfer_cache.py`
- `Engine/RetrievalAttention/cache_hub/flash_attn_cache.py`
- `Engine/RetrievalAttention/attn_hub/retroinfer_attn.py`
- `Engine/RetrievalAttention/attn_hub/flash_attn.py`
- `Engine/RetrievalAttention/benchmark/config.py`

## Current 3-Stage Algorithm

### Stage definitions
- Stage 1 (Draft): `engine.speculate(...)`
  - Attention type: `RetroInfer`
  - KV budget: `budget1`
  - Generates `gamma1` tokens (plus internal bonus-token mechanics)
- Stage 2 (Early verify): `engine.verify(...)`
  - Attention type: `RetroInfer`
  - KV budget: usually `budget2`, dynamically adjusted to `budget2_low` or `budget2_high`
  - Verifies draft outputs; may be skipped when confidence is high
- Stage 3 (Settle): `engine.settle(...)`
  - Attention type: `Full_Flash_Attn` (full KV, effectively 100%)
  - Final authority: only tokens matching settle outputs survive

### Outer control loop
For each sample:
1. Prefill-like bootstrap via `engine.encode(input_ids)`.
2. Repeat speculate/verify cycles until either:
   - unsettled token count reaches `gamma2`, or
   - verify-call guard triggers, or
   - EOS is reached.
3. Run settle, accept shared prefix with settle output, discard rest.
4. Continue until `num_gen_token_max` or EOS.

### Dynamic budget logic
In `run_3step_profile_optimized.py`:
- Confidence metric is `top1_top2_diff` from draft stage.
- If `min_confidence > confidence_threshold`:
  - Use lower verify budget (`budget2_low`), and skip explicit verify compute (`pass_verify=True`).
- If `min_confidence < confidence_threshold_low`:
  - Use higher verify budget (`budget2_high`).
- Else use default `budget2`.
- Budget update path: `engine.update_verification_budget(...)` mutates RetroInfer verification config (`nprobe`, `cache_cluster_num`, `max_compute_cluster_num`).

## Important Implementation Reality (Current)

### `generate_without_prefill_token` still performs prefill
In `model_hub/LLM.py`:
- `generate_without_prefill_token(...)` calls:
  - `init_kv_cache(...)`
  - `inference_without_prefill_token(...)`
- `inference_without_prefill_token(...)` does:
  - `_ = self.prefill_forward(inputs_ids=inputs_ids)`
  - then decode from `bonus_token`.

Therefore, each call to `speculate()` and `verify()` reinitializes KV cache and re-runs prefill for the whole current prefix.

### `settle()` also reprefills
`backend_for_3stage.py::settle()` calls `model.generate(...)`, and `generate(...)` also allocates a fresh cache and performs full prefill each call.

## Where latency is wasted
Redundant prefill occurs on every inner-loop stage call:
- Draft: each speculate call
- Verify: each verify call
- Settle: each settle call

Given long prefixes, prefill dominates latency. Even if logic is functionally correct, this architecture is not suitable for real online inference where prefill should run once per request (or once per committed frontier).

## KV/Attention internals relevant to optimization

### RetroInfer cache behavior
`retroinfer_cache` maintains multiple structures:
- `steady_zone_keys/values` with append behavior via `static_pattern_total`
- centroid/index metadata for retrieval
- per-layer compute buffers for decode

Decode appends one token per step via `decode_update_kv_cache()` and uses `compute()` for retrieval attention.

### Full attention cache behavior
`flash_attn_cache` maintains contiguous key/value arrays with append position from `valid_length`.

### Budget adaptation is metadata-level
Budget switches primarily alter retrieval configuration (`nprobe` and related counts), not the model weights. This suggests dynamic verify budget can be done without rebuilding full request-prefill state.

## Compatibility constraints
Optimization must preserve:
1. Existing CLI args and defaults in `run_3step_profile_optimized.py`.
2. Existing dynamic budget decision policy and logging semantics.
3. Existing token acceptance logic (draft/verify/settle).
4. Existing EOS and termination behavior.
5. Existing CSV and `stage_outputs.json` outputs (format-compatible).

## Refactoring freedom (important)
- You may create new Python files when this leads to a cleaner design.
- Prefer a clean new runner/backend module over turning existing files into spaghetti code.
- Example: creating a new optimized runner (instead of heavily patching `tests/dynamic_verify_test/run_3step_profile_optimized.py`) is explicitly allowed.
- Keep backward compatibility by preserving the old path and adding a clear migration/fallback path.

## Proposed Optimization Design

### Design objective
Avoid repeated prefill by introducing persistent request-level decode sessions and branch snapshots.

### Core idea
Use a sessionized backend with persistent caches:
- One persistent RetroInfer session for speculative path (draft + early verify).
- One persistent Full-FLASH session for settled path (final authoritative path).
- Maintain a committed frontier and branch frontier through snapshot/restore, instead of rebuilding cache from scratch.

### Recommended API additions (high level)
In `LLM` and cache layers, add session APIs such as:
- `begin_session(attention_type, prompt_ids, attention_mask, attn_config, ...)`
  - Performs prefill once, initializes persistent kv state.
- `decode_tokens(session, bonus_or_input_tokens, num_new_tokens, ...)`
  - Decodes incrementally from existing cache.
- `snapshot(session)` and `restore(session, snapshot)`
  - Save/restore lightweight cache state pointers + metadata needed for rollback.
- `update_session_attn_config(session, attn_config_delta)`
  - Support dynamic verify budget changes without session re-init.
- `end_session(session)`

### Branching model
Maintain two frontiers:
- Settled frontier (authoritative, full-attn session).
- Verified frontier (temporary speculative frontier, retroinfer session).

Workflow:
1. Start both sessions once from initial prompt.
2. For each speculate/verify micro-cycle:
   - Snapshot verified session state.
   - Draft decode on retroinfer session (budget1).
   - Verify decode on same retroinfer session (budget2 dynamic) or skip-verify fast path.
   - If tokens are rejected later by settle, restore verified session to snapshot/committed position.
3. When settle accepts prefix, append accepted tokens to settled session and advance committed frontier.
4. Reset verified frontier to committed frontier (by restore/fast-forward), not by reprefill.

### Rollback considerations
Rollback must reset, at minimum:
- logical sequence length/context pointers
- static-pattern cursor(s)
- any per-step counters affecting retrieval behavior (example: first-token sharing counters)
- if index update thresholds are crossed, ensure deterministic rollback semantics

If full rollback of some internals is expensive, use bounded snapshot interval and deterministic replay from nearest checkpoint (still avoiding full prefix prefill).

## Implementation strategy (phased)

### Phase 1: Session scaffolding, no behavior change
- Add session objects and wrappers.
- Keep old `generate*` paths untouched.
- Add feature flag (example: `--use_persistent_kv`). Default off.

### Phase 2: Move draft/verify to persistent retroinfer session
- Eliminate repeated prefill from speculate/verify loops.
- Keep settle path unchanged initially.

### Phase 3: Move settle to persistent full-attn session
- Eliminate repeated prefill from settle.
- Finalize snapshot/restore semantics between verified and settled frontiers.

### Phase 4: Enable-by-default after validation
- Keep legacy path for fallback/regression triage.

## Verification Plan (must be run)

### A. Functional equivalence
For fixed seed and identical args:
1. Compare generated token IDs per example between old and new implementations.
2. Compare acceptance counts per stage:
   - verify accepted count per call
   - settle accepted count per call
3. Compare termination condition triggers (EOS vs max-gen).

Acceptance criterion:
- Exact match in token IDs for deterministic settings.
- If numerical nondeterminism appears, define strict tolerance policy and document causes.

### B. Logging compatibility
Ensure outputs remain compatible:
- `step_log.csv` columns and meaning unchanged.
- `accumulated_log.csv` columns and meaning unchanged.
- `stage_outputs.json` schema unchanged.

### C. Performance validation
Measure and compare:
1. Total runtime per step and end-to-end runtime.
2. Aggregate prefill latency share before/after.
3. Stage call latency distributions.

Expected outcome:
- Significant reduction in end-to-end latency for long prefixes, especially many speculate/verify loops.

### D. Memory safety
Stress tests for long prefixes and many cycles:
- no unbounded growth of temporary tensors
- no stale-cache contamination across samples
- cleanup still works

### E. Backward compatibility and fallback
- Old path remains runnable and unchanged with flag off.
- New path can be disabled quickly for troubleshooting.

## Known risk areas
1. Correct rollback of RetroInfer internal retrieval metadata after token rejection.
2. Consistency of `static_pattern_end`/cursor logic when mixing accepted and discarded tokens.
3. Interaction of `use_first_kv` token counters with snapshot restore.
4. Determinism differences due to changed execution order/state reuse.

## Deliverable requirements for implementation PR
1. New session-based API in model/cache layers.
2. Backend changes to use persistent sessions.
3. No CLI-breaking changes.
4. Regression script comparing old/new token traces and logs.
5. Benchmark script/report demonstrating reduced prefill overhead.
6. Clear fallback switch to legacy behavior.
