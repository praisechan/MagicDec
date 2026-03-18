You are GPT-Codex acting as a senior systems engineer for speculative decoding in MagicDec.

Objective:
Implement RetrievalAttention_new-based self-speculative decoding support with a new backend and a new 3-stage hierarchical speculative benchmark flow.

Latest Amendment (must follow in current version):
- Online loop must accumulate survived online tokens from Draft+Early Verify until gamma2 (or EOS/max-token boundary), then perform one authoritative final-settlement comparison with a live Final Verify call over that pending span.
- `final_verify_calls` in per-step/accumulated stats should represent these chunk-level settlement events (not every Early Verify iteration).
- Keep a persistent final_verify cache during online decoding; do not use one-time pre-run groundtruth-only settlement.
- For `engine.commit_prefix("draft", ...)`, use budget2 behavior during replay only, then continue normal `engine.speculate(...)` with budget1 behavior.
- Do not directly rely on Python-side runtime mutation of `nprobe`/related values as a standalone mechanism, because low-level WaveBuffer state can diverge; if budget switching is needed, use a safe backend routing/synchronization strategy or implement explicit low-level setter support.
- When synchronizing committed growth across RetroInfer caches, include retrieval metadata (centroids/value_sum/centroids_mask/cluster_size) in addition to steady-zone KV/length state when shapes are compatible.

Critical Rules:
1. Do not reuse or reference any previous text prompt.
2. Before any code changes, you must first read these files:
  - MagicDec/tests/StreamingLLM/selfspec_benchmark.py
  - MagicDec/Engine/RetrievalAttention_new/model_hub/llama.py
  - MagicDec/Engine/RetrievalAttention_new/model_hub/LLM.py
  - MagicDec/Engine/RetrievalAttention_new/cache_hub/retroinfer_cache.py
  - MagicDec/Engine/RetrievalAttention_new/cache_hub/flash_attn_cache.py
  - MagicDec/Engine/RetrievalAttention_new/** (focus on cache_hub, config, model_hub, benchmark/longbench)
  - MagicDec/Engine/RetrievalAttention_new/benchmark/longbench/pred.py
3. Before proceeding to code edits, you must explicitly trace and understand function call stacks step by step for:
  - MagicDec/tests/StreamingLLM/selfspec_benchmark.py (including downstream backend/model/cache calls)
  - MagicDec/Engine/RetrievalAttention_new/benchmark/longbench/pred.py (including model loading, config generation, generation path, and cache/attention-related calls)
  - MagicDec/Engine/RetrievalAttention_new/model_hub/llama.py, model_hub/LLM.py, cache_hub/retroinfer_cache.py, cache_hub/flash_attn_cache.py (target implementation path)
  - You must use this call-stack understanding to design RetrievalAttention_new backend behavior.
4. You must not refer to or copy logic from these old/incorrect paths:
   - MagicDec/Engine/RetrievalAttention
   - MagicDec/tests/dynamic_verify_test
   - MagicDec/tests/RetrievalAttention
   - RetrievalAttention_old
5. Preserve existing coding style and avoid breaking current StreamingLLM/SnapKV behavior.
6. Main technical focus is KV cache maintenance correctness across stages.

Execution Environment Requirements:
1. Before running any code or tests, run: 
  conda activate retroinfer
  export PYTHONPATH=/home/juchanlee
2. Run all Python executions on GPU 3 of the shared lab server by setting:
  - CUDA_VISIBLE_DEVICES=3
3. For all benchmark/test executions, enforce a timeout to avoid stuck runs (for example via timeout command).
4. When running script, show the output of terminal to me, not using redirection ">".

Work Item 1: Add RetrievalAttention_new engine backend for self-spec
- Add necessary methods in backend file at:
  - MagicDec/Engine/RetrievalAttention_new/backend.py
- Backend must be callable from a new benchmark script at:
  - MagicDec/tests/RetrievalAttention_new/selfspec_benchmark.py
- Match the practical interface pattern used by existing self-spec engines (StreamingLLM/SnapKV style):
  - load_model(...)
  - setup_caches(...)
  - encode(...)
  - speculate(...)
  - verify(...)
  - clear_kv(...)
- For RetrievalAttention_new integration, ensure attention mode and budget are configurable and switchable per stage.
- Ensure cache-length/page-table updates are explicit and auditable after each stage call.

Work Item 2: Implement 3-stage hierarchical speculative decoding
Stage API contract:
1. Stage 1 Draft:
   - engine.speculate(...)
   - attention type: RetroInfer
   - KV budget: budget1
  - generate gamma1 tokens autoregressively
2. Stage 2 Early Verify:
   - engine.early_verify(...)
   - attention type: RetroInfer
   - KV budget: budget2 (with optional dynamic adjustments if you keep that feature)
  - verify gamma1 drafted tokens in one step and compute accepted span
  - this stage produces a bonus token at the end; bonus-token cache/token accounting is required
3. Stage 3 Final Verify (Settle):
   - engine.final_verify(...)
   - attention type: Full_Flash_Attn
   - full KV budget (effectively 100%)
  - keep this cache live during iterative decoding
  - invoke this stage at each gamma2 settlement boundary (or EOS/max-token boundary)
  - verify pending online span in one call and use final output as authoritative settlement reference
  - this stage contributes authoritative bonus-token behavior for accounting consistency

Detailed implementation guideline (must follow):
1. Final Verify stage live settlement:
  - Initialize final_verify cache once after prefill with Full_Flash_Attn and full KV budget.
  - During iterative decoding, invoke `engine.final_verify(current_final_token, pending_online_tokens)` at each settlement boundary.
  - Use this live output as the authoritative reference for settlement.
  - Revert final_verify cache to chunk-start snapshot before settlement replay, then commit the authoritative accepted span to keep all caches aligned.

2. KV cache maintenance simplification:
  - Maintain three KV cache objects:
    - `draft_kv_cache` for Stage 1 draft generation.
    - `early_verify_kv_cache` for Stage 2 early verification.
    - `final_verify_kv_cache` for Stage 3 authoritative settlement.
  - Implement explicit KV snapshot/revert capability for both draft and early-verify caches.
  - Required cache API behavior:
    - `snapshot_state()` or equivalent: capture current cache length/page-table pointers.
    - `revert_to(snapshot)` or equivalent: restore cache to exact prior state.
    - `truncate_to(length)` is acceptable if mathematically equivalent and fully auditable.
  - Revert rule after early verify:
    - Draft stage appends `gamma1` tokens to `draft_kv_cache`.
    - If only `accepted_len < gamma1` are accepted, revert `draft_kv_cache` to remove discarded suffix tokens.
    - Apply the same revert logic to `early_verify_kv_cache` so both stage caches stay aligned on committed prefix.
  - Revert rule at final settlement:
    - Revert draft/early/final caches to chunk-start snapshots.
    - Commit authoritative accepted span from final verify to final cache, then synchronize into early and draft caches.
    - Ensure synchronized replay carries retrieval metadata where applicable.

3. Bonus token handling (strict):
  - Bonus token is always present from a verify call output shape, regardless of full acceptance.
  - Early verify returns `gamma1 + 1`-style outputs conceptually: compared prefix plus one bonus token.
  - Accepted continuation for next draft is:
    - accepted draft prefix up to first mismatch (possibly full `gamma1`), then
    - the early-verify bonus token.
  - Example A (`gamma1 = 4`):
    - draft tokens: `[24, 64, 76, 87]`
    - early verify outputs: `[24, 64, 76, 87, 98]`
    - accepted draft tokens: `[24, 64, 76, 87]`
    - bonus token: `98`
    - next committed extension: `[24, 64, 76, 87, 98]`
  - Example B (`gamma1 = 4`):
    - draft tokens: `[24, 64, 76, 87]`
    - early verify outputs: `[24, 64, 56, 31, 12]`
    - accepted draft tokens before mismatch: `[24, 64]`
    - bonus token (at mismatch position): `56`
    - next committed extension: `[24, 64, 56]`
  - Call-count accounting must still count verify invocations that produce only bonus extension after partial match.

4. Groundtruth alignment policy with live final verify:
  - At settlement, compare pending online span against the same-length prefix of live final_verify outputs.
  - If mismatch is detected, treat final_verify token at mismatch position as authoritative settle token.
  - Continue next chunk from the authoritative final token state and synchronized draft/early caches.

Hierarchical loop contract (must be implemented exactly):
1. Run Draft (gamma1 autoregressive generation).
2. Run Early Verify once to verify that gamma1 block.
3. Commit accepted tokens plus one early-verify bonus token.
4. Run live Final Verify at settlement boundaries for authoritative comparison/settlement.
5. Repeat Draft -> Early Verify -> Groundtruth alignment until termination condition (EOS or max_new_tokens).

Required implementation behavior:
- Terminal output must include per-stage measurements for every iteration:
  - elapsed time per stage
  - accepted tokens
  - rejected tokens
  - cumulative unsettled tokens (if using unsettled flow)
  - budget used in each stage
  - actual detokenized sentence in natural language generated by draft, early verify, final verify stage
- Maintain clear token accounting:
  - drafted tokens count
  - early-verified accepted count
  - final settled token count (authoritative committed span)
  - per-stage bonus tokens (early verify bonus, final verify-derived authoritative bonus)
  - inference call counts must reflect bonus-token-producing verify calls correctly
  - emitted tokens
- Ensure EOS handling is consistent and does not leave cache/state inconsistent.
- Ensure rollback/re-advance logic for cache lengths is mathematically consistent after partial acceptance.
- Ensure cache revert operations are executed for both draft and early-verify caches whenever discarded tokens occur.

Output files and logging format (must match run_3step_profile_highlow_extreme.py style)
Create logging directory and output files:
- step_log.csv
- accumulated_log.csv
- stage_outputs.json

CSV schema requirements:
1. step_log.csv headers (exact order):
   - step
   - dataset
   - prefix_len
   - gamma1
   - gamma2
   - budget1
   - budget2
   - speculate_calls
   - early_verify_calls
   - final_verify_calls
   - tokens_generated

2. accumulated_log.csv headers (exact order):
   - step
   - dataset
   - prefix_len
   - gamma1
   - gamma2
   - budget1
   - budget2
   - total_speculate_calls
   - total_early_verify_calls
   - total_final_verify_calls
   - total_tokens_generated

3. stage_outputs.json format:
- Append per-stage entries with fields:
  - stage: one of draft, early verify, final verify
  - outputs: token outputs list

Required console summary format:
- Per step print section:
  - === Step {step} Statistics ===
  - Dynamic budget enabled: ...
  - Speculate calls: ...
  - Early Verify calls: ...
  - Final Verify calls: ...
  - Tokens generated: ...
- Running totals print section:
  - === Accumulated Statistics (up to step {step}) ===
  - Total speculate calls: ...
  - Total early verify calls: ...
  - Total final verify calls: ...
  - Total tokens generated: ...
- Final totals print section:
  - === Final Accumulated Statistics ===
  - Total speculate calls: ...
  - Total early verify calls: ...
  - Total final verify calls: ...
  - Total tokens generated: ...

Implementation boundaries:
- Use RetrievalAttention_new internals for model/config/cache behavior.
- Do not import from old RetrievalAttention directories.
- Keep old benchmark scripts runnable.
- Prefer adding new files rather than mutating unrelated existing pipelines.

Verification checklist you must run before finishing:
1. Static checks for syntax/import errors on newly added files.
2. Progressive verification with gradually increasing scope (each step with timeout and CUDA_VISIBLE_DEVICES=3 under conda env retroinfer):
  - prefill stage only
  - prefill + single draft stage
  - prefill + draft + early verify
  - prefill + draft + early verify + final verify
  - multi-sample run covering the full 3-stage loop
3. For each progressive step, confirm program does not hang and completes within timeout budget.
4. Confirm logs are written to all three output files.
5. Confirm stage_outputs.json contains draft/early verify/final verify entries.
6. Confirm accumulated_log.csv contains a final aggregated row.
7. Confirm bonus-token accounting is visible in per-stage logs and consistent with inference call counters.

Deliverables:
1. New file: MagicDec/tests/RetrievalAttention_new/selfspec_benchmark.py
2. Any minimal support edits required for imports/initialization.
3. A concise implementation summary listing:
   - cache state variables per stage
   - acceptance/rejection decision flow
  - gamma1/gamma2 hierarchical loop invariants
  - bonus-token handling and how it affects token/call counts
   - where each metric is computed and logged
