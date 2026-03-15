# Prompt for Claude Opus: Implement Persistent-KV Optimization for 3-Stage Hierarchical Self-Speculative Decoding

You are working in the MagicDec codebase. Implement a performance optimization for 3-stage hierarchical self-speculative decoding by removing redundant prefill operations while preserving behavior and compatibility.

## Environment setup
- Use the `retroinfer` conda environment before running code:
   - `conda activate retroinfer`
- If you modify `library/retroinfer`, reinstall it so changes are reflected:
   - `cd library/retroinfer && pip install .`

## Context
Current runner:
- `tests/dynamic_verify_test/run_3step_profile_optimized.py`

Current backend:
- `Engine/RetrievalAttention/backend_for_3stage.py`

Core model stack:
- `Engine/RetrievalAttention/model_hub/LLM.py`
- `Engine/RetrievalAttention/model_hub/llama.py`
- `Engine/RetrievalAttention/model_hub/qwen.py`
- `Engine/RetrievalAttention/cache_hub/retroinfer_cache.py`
- `Engine/RetrievalAttention/cache_hub/flash_attn_cache.py`
- `Engine/RetrievalAttention/benchmark/config.py`

Important detail:
- `generate_without_prefill_token()` still does prefill internally (`init_kv_cache` + `prefill_forward`) and then discards the auto-generated token behavior by forcing bonus-token decode semantics.
- As a result, `speculate()`, `verify()`, and `settle()` repeatedly reprefill and are slow.

## Goal
Refactor to run prefill only once per request/session (or once per committed frontier) and reuse KV cache for subsequent draft/verify/settle decoding.

## Constraints
1. Preserve functional behavior of current decoding logic.
2. Preserve CLI compatibility for existing scripts and arguments.
   - Compatibility here specifically means: previous code path must still be runnable with the same terminal command as before.
3. Preserve output formats of:
   - `step_log.csv`
   - `accumulated_log.csv`
   - `stage_outputs.json`
4. Keep a legacy fallback path available (feature flag or parallel code path).
5. Do not degrade correctness of dynamic verification budget logic.
6. You are allowed to create new Python files for a cleaner implementation.
7. Prefer clean modular files over heavily editing existing spaghetti-style scripts.
8. You may create new files/classes/modules as needed for a clean implementation.
9. Do not build the new optimized path on top of `tests/dynamic_verify_test/run_3step_profile_optimized.py`.
   - Treat that script as legacy/reference only; implement and wire the new path through a separate runner/module.

## Required behavior to preserve
- Stage 1 Draft uses `RetroInfer` with `budget1`.
- Stage 2 Early Verify uses `RetroInfer` with dynamic budget (`budget2` / `budget2_low` / `budget2_high`) based on confidence.
- Stage 3 Settle uses full attention (`Full_Flash_Attn`) as final authority.
- Token acceptance/rejection semantics must remain unchanged.
- EOS and max-token termination conditions must remain unchanged.

## Implementation approach (recommended)

### 1. Add sessionized inference API
In `LLM.py` and model/cache integration, add persistent session methods so cache is initialized once and reused:
- begin/init session with prefill once
- incremental decode calls without cache re-init
- lightweight snapshot/restore for rollback
- update attention config (especially RetroInfer nprobe/cache params) without full restart

Keep existing `generate()` and `generate_without_prefill_token()` intact for backward compatibility.

### 2. Add persistent state in backend
In `backend_for_3stage.py`:
- introduce request/session lifecycle methods
- replace repeated per-call `generate*` usage in `speculate`, `verify`, `settle` with session decode calls
- maintain two frontiers:
  - settled frontier (full-attn authoritative)
  - verified/speculative frontier (retroinfer)
- implement snapshot/restore around speculate/verify loops so rejected tokens do not require re-prefill

### 3. Dynamic budget update support
When budget changes for verify stage:
- update RetroInfer retrieval config in-session (nprobe, cache_cluster_num, max_compute_cluster_num)
- avoid rebuilding full prefix cache

### 4. Legacy fallback
Add runtime switch (for example `use_persistent_kv`) defaulting to legacy-safe behavior initially, then easy to flip for benchmarking.

## Critical risk handling
1. Rollback correctness for RetroInfer internal state:
   - context counters
   - static-pattern pointers
   - any per-layer counters (for example first-token sharing counters)
2. Ensure state after settle commit exactly matches accepted prefix.
3. Ensure no stale data from discarded branch affects next cycle.
4. Maintain deterministic behavior as much as possible.

## Verification requirements (must implement)

### A. Functional parity checks
Build a regression harness that runs old and new paths with same seed/args and compares:
1. Generated token IDs per sample
2. Verify accepted token counts per call
3. Settle accepted token counts per call
4. Termination reason (EOS or max-token)

If exact parity is not guaranteed due to numerical nondeterminism, quantify divergence and document why.

### B. Logging parity checks
Ensure CSV/JSON schema compatibility and semantically equivalent values for counters.

### C. Performance checks
Benchmark old vs new and report:
1. End-to-end runtime per request
2. Stage-level latency breakdown
3. Prefill time share reduction

### D. Memory checks
Run multi-step stress cases and verify:
- no memory leaks
- no OOM regression
- cleanup path still works

### E. Detailed step-by-step verification flow (must follow in order)
Add explicit instrumentation and execute verification in this exact sequence:
1. Prefill-stage token trace parity:
   - Verify token IDs generated immediately after the first prefill stage are correct.
   - Implement code that records token trace artifacts for both legacy path and new persistent-KV path.
   - Compare traces side by side for the same input, seed, and arguments.
2. First draft-stage parity:
   - Verify the first draft stage generates the same token IDs as the original code.
3. Early-verify accept/reject boundary parity:
   - Verify early-verify accepts and rejects at the same token positions as the original code.
4. Multi-cycle parity before final settle:
   - Run longer repetitions of (draft -> early verify) cycles before final verification.
   - Confirm parity trends hold across repeated cycles, not only on the first cycle.
5. Final settle-stage comparison:
   - Compare final verification/settle outputs between old and new paths.
   - Report where outputs are identical and where they diverge.

### F. Handling and explaining non-identical results
When results differ, follow this policy:
1. First, attempt to make token-ID traces match the original code as much as possible.
2. Only after careful debugging, consider the known legacy inconsistency as a possible cause:
   - The original code may be slightly incorrect because draft and early-verify stages effectively use KV cache produced from repeated full prefill (100% context) rather than strictly decode-only KV growth after one prefill.
   - The intended experiment here is: prefill once, then grow KV only via decode with partial KV usage.
   - Because legacy repeatedly re-prefills per stage, token IDs may differ from the persistent-KV design.
3. Do not assume this explanation by default.
   - Before attributing divergence to legacy behavior, rule out other bugs (rollback bugs, stale cache reuse, budget update errors, frontier mismatch, or state contamination).
4. In the final report, clearly separate:
   - confirmed implementation bugs
   - expected/understood divergences caused by differing KV semantics

## Deliverables
1. Code changes implementing persistent-KV decoding with rollback/snapshot support.
2. Legacy-compatible execution path.
3. Regression/benchmark scripts or commands.
4. Short report in markdown summarizing:
   - design decisions
   - parity results
   - speedup numbers
   - known limitations
5. If new files are introduced, include a brief file map and rationale for why each new file improves maintainability.

## Coding quality expectations
- Keep edits minimal and localized where possible.
- Add concise comments for non-obvious cache/session transitions.
- Do not break existing public APIs unless adding optional params.
- Include clear error handling for invalid rollback/snapshot states.

## Output format for your final response
When done, provide:
1. Summary of changed files.
2. Explanation of new session lifecycle and rollback model.
3. Exact commands used for regression and benchmark.
4. Measured parity and performance results.
5. Any residual risks and next steps.
