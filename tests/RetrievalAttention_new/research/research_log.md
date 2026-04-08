## [Phase 1.0] Initial codepath reading and mechanism trace — 2026-04-05 17:42:52 KST
### What was done
- Read the required files in the requested order:
- `prompts/retrievalattention_new_3stage_implementation_summary.md`
- `Engine/RetrievalAttention_new/backend.py`
- `tests/RetrievalAttention_new/selfspec_benchmark.py`
- `Engine/RetrievalAttention_new/cache_hub/retroinfer_cache.py`
- `Engine/RetrievalAttention_new/model_hub/llama.py`
- Traced the active decode path further into:
- `Engine/RetrievalAttention_new/model_hub/LLM.py`
- `Engine/RetrievalAttention_new/attn_hub/retroinfer_attn.py`
- `Engine/RetrievalAttention_new/attn_hub/full_attn.py`
- Confirmed this is a fresh research workspace: `tests/RetrievalAttention_new/research/` did not exist and was created.
- Checked local workspace state to avoid clobbering unrelated edits.

### Key findings
- `_activate_cache` in `backend.py` only swaps `model.attention_type` and `model.kv_cache`; all stages share the same model weights and `decode_forward` path.
- Full attention uses `flash_attn_with_kvcache` over the whole live KV cache, while RetroInfer decode uses `retroinfer_cache.attn_func(...)`.
- RetroInfer keeps a `steady_zone` exactly and compresses the long-range region into centroid/value-sum metadata built by segmented k-means.
- At decode time RetroInfer does:
- score query against centroids,
- select top `nprobe` clusters plus an optional estimation zone,
- gather retrieved pages from CPU/GPU cache into an execution buffer,
- run attention over `steady_zone + retrieved pages`, optionally merged with estimation-zone summary outputs.
- Therefore the lost information versus full attention is not just "lower confidence"; it is omission of non-selected KV pages and replacement of some cluster contents by coarse centroid/value-sum summaries.
- Early verify acceptance only means per-position argmax agreement with the drafted token sequence up to the accepted span; the probability shape can still differ materially.
- Final verify rejection happens when full attention changes the first mismatch position within the pending span; that means the missing or distorted KV evidence was decisive enough to flip the authoritative next-token argmax.

### Open questions
- Whether rejected positions are dominated by retrieval misses, distribution fragility, positional drift, or a mixture.
- How separable rejected positions are using pre-final signals such as divergence between draft and early verify, budget sensitivity, or hidden-state drift.
- Whether bonus-token failures differ materially from drafted-token failures.

### Next step
- Write dedicated Phase 2 probing scripts for:
- logit/distribution inspection,
- hidden-state comparison,
- budget-sensitivity analysis,
- then run a small smoke test to validate the instrumentation path.

## [Phase 2.1] Probe implementation and smoke validation — 2026-04-05 18:02:00 KST
### What was done
- Added new experimental files under `tests/RetrievalAttention_new/research/`:
- `probe_utils.py`
- `phase2b_logit_probe.py`
- `phase2c_hidden_state_probe.py`
- `phase2d_budget_sensitivity.py`
- Added `scripts.md` entries for all new probes.
- Verified syntax with `python -m py_compile` in the `retroinfer` conda env.
- Identified the correct runtime environment for this repo’s GPU stack:
- `conda run -n retroinfer python ...`
- Ran live smoke tests on GPU 1 with a shortened config:
- `phase2b_logit_probe.py` with `prefix_len=2048`, `gamma1=2`, `gamma2=4`, `num_max_token=6`, `max_cycles=1`
- `phase2c_hidden_state_probe.py` with the same reduced settings
- `phase2d_budget_sensitivity.py` with the same reduced settings

### Key findings
- All three probe scripts executed successfully against the real 8B model and RetrievalAttention caches.
- Smoke-test artifacts were created:
- `tests/RetrievalAttention_new/research/data/phase2b_logit_probe_smoke.csv` with 7 rows
- `tests/RetrievalAttention_new/research/data/phase2c_hidden_state_probe_smoke.csv` with 3 rows
- `tests/RetrievalAttention_new/research/data/phase2d_budget_sensitivity_smoke.csv` with 7 rows
- A real probe bug surfaced and was fixed:
- final settlement can include positions with no same-cycle draft/early logits, especially the authoritative bonus position,
- probe scripts now treat those positions as unavailable instead of incorrectly indexing shorter tensors.
- The default shell `python` is unusable for this work because it lacks `torch`; the `retroinfer` env is required.

### Open questions
- The probes are validated only on tiny smoke runs, not yet on the target 32k-prefix research configuration.
- Need actual Phase 2 data before testing any hypotheses about rejection mechanism.
- Need to confirm that the encoded full-logit/full-softmax CSV payloads remain tractable at 32k / 10-20 settlement cycles.

### Next step
- Run `phase2b_logit_probe.py` with the requested research settings on GPU 1 using the `retroinfer` env, starting with `num_eval_steps=1` and `max_cycles=10`, then inspect the resulting CSV for accepted vs rejected separation patterns.

## SESSION HANDOFF — 2026-04-05 18:02:00 KST
### Current phase and sub-step
Phase 2b/2c/2d setup complete; probe scripts implemented and smoke-tested, full Phase 2 experiments not yet run.

### State of work
- Phase 1 reading and summary completed.
- Research scaffold created.
- Three new Phase 2 probing scripts exist and run successfully on reduced settings.
- No large-scale experimental data or phase summaries beyond Phase 1 yet.

### Critical context the next session MUST know
- Use `conda run -n retroinfer python ...`; the default `python` in base env does not have `torch`.
- `_activate_cache` only swaps cache objects; the same model path is reused across stages, so mechanistic differences come from cache-attention behavior, not model weights.
- RetroInfer approximation error is structural: selected clusters + steady zone + optional estimation summaries replace full token-level KV access.
- Smoke tests revealed that final settlement may include positions with no corresponding same-cycle early/draft logits; the probes now explicitly allow missing early/draft payloads at those positions.

### Exact next action
- Run:
- `CUDA_VISIBLE_DEVICES=1 conda run -n retroinfer python tests/RetrievalAttention_new/research/phase2b_logit_probe.py --dataset pg19 --model_name Meta-Llama-3.1-8B --prefix_len 32768 --num_max_token 100 --num_eval_steps 1 --max_cycles 10 --gamma1 6 --gamma2 32 --budget1 0.02 --budget2 0.10 --budget2_high 0.20 --enable_dynamic_budget --T_low 0.05 --T_high 0.2`
- Then inspect the output CSV size, row counts, and a few rejected vs accepted rows before launching the companion 2c and 2d runs.

### Files to read first
- `tests/RetrievalAttention_new/research/research_log.md`
- `tests/RetrievalAttention_new/research/phase1_summary.md`
- `tests/RetrievalAttention_new/research/probe_utils.py`
- `tests/RetrievalAttention_new/research/phase2b_logit_probe.py`

## [Phase 2.2] Real 32k probe runs and synthesis — 2026-04-05 19:20:53 KST
### What was done
- Ran the requested real Phase 2b logit probe in the `retroinfer` env:
- `CUDA_VISIBLE_DEVICES=1 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase2b_logit_probe.py --dataset pg19 --model_name Meta-Llama-3.1-8B --prefix_len 32768 --num_max_token 100 --num_eval_steps 1 --max_cycles 10 --gamma1 6 --gamma2 32 --budget1 0.02 --budget2 0.10 --budget2_high 0.20 --enable_dynamic_budget --T_low 0.05 --T_high 0.2`
- Inspected the generated CSV for row count, cycle count, rejection count, mismatch positions, and accepted-vs-rejected distribution statistics.
- Ran the companion Phase 2c hidden-state probe with the same main configuration.
- Ran the companion Phase 2d budget-sensitivity probe with the same main configuration.
- Compared mismatch positions and nearby positions across 2b/2c/2d outputs.

### Key findings
- Phase 2b artifact: `tests/RetrievalAttention_new/research/data/phase2b_logit_probe_Meta-Llama-3.1-8B_32768.csv`
- Size: 54 MB, 137 rows across 5 settled cycles.
- Rejections: 2 rejected cycles, 3 accepted cycles; mismatch positions were 9 and 16; neither rejection was on the bonus token.
- Dynamic mode usage in 2b skewed heavily toward `high`: 91 rows `high`, 46 rows `normal`, 0 rows `skip`.
- Important instrumentation caveat: only 35/137 rows still had same-cycle early logits and 30/137 had same-cycle draft logits. The actual mismatch rows in this run had no aligned early/draft payload because the final settled span extended beyond the latest local verify window.
- Within the 35 rows that did have early logits, rejected-block positions showed much worse early-vs-final agreement than accepted-block positions:
- rejected-block rows with early logits: 13/14 had top-10 overlap = 0, 0/14 had final top-1 ranked in early top-10, mean overlap 0.07, median final-top1 rank 976, mean KL(final||early) 11.17.
- accepted-block rows with early logits were bimodal: 7/21 had overlap >= 8 and JS < 0.01, but 10/21 still had overlap = 0 and 10/21 had final-top1 rank > 1000.
- This means "distribution similarity" is a real mechanism, but not a clean single-threshold signal yet. Some accepted positions remain highly distribution-shifted even when argmax agrees.
- Phase 2c artifact: `tests/RetrievalAttention_new/research/data/phase2c_hidden_state_probe_Meta-Llama-3.1-8B_32768.csv`
- Size: 42 rows across 6 cycles, 4 rejected cycles.
- Hidden-state drift separated accepted vs rejected blocks more clearly than logit overlap:
- accepted-block positions: cosine mean 0.639, L2 mean 91.3.
- rejected-block positions: cosine mean 0.325, L2 mean 163.0.
- The one directly observed rejected position had cosine 0.190 and L2 179.45; its left neighbor was even lower cosine (0.079) with L2 194.39.
- This supports a mechanistic picture where retrieval approximation error distorts the representation before the first argmax flip, with drift already visible in the positions immediately preceding rejection.
- Phase 2d artifact: `tests/RetrievalAttention_new/research/data/phase2d_budget_sensitivity_Meta-Llama-3.1-8B_32768.csv`
- Size: 157 rows across 6 cycles, 3 rejected cycles; mismatch positions were 8, 9, and 3.
- Budget sensitivity was weaker than expected for modest budget changes (0.08 / 0.10 / 0.12):
- rejected positions: 0/3 argmax changes across budgets, mean margin range 0.0145, mean entropy range 0.0335.
- accepted positions: 2/78 argmax changes across budgets, mean margin range 0.0178, mean entropy range 0.0368.
- At all three rejected positions, all three budgets produced the same wrong early argmax; increasing budget from 0.10 to 0.12 did not recover the full-attention token.
- Therefore the dominant failure mode in these sampled rejections is not "borderline around budget2". It looks more like a stable retrieval miss or approximation distortion that persists across small budget adjustments.

### Open questions
- Phase 2b currently under-observes the actual mismatch rows at the early/draft-logit level because settlement spans can include positions beyond the latest local verify window.
- Need Phase 3 logging to align draft/early features to the full settled span so that mismatch positions themselves always have pre-final features attached.
- Need to test whether stronger budget changes or explicit retrieval-page diagnostics reveal recovery on rejected positions; the 20% local budget increase used here was not enough.
- Need to quantify whether hidden-state drift can be approximated by a pre-final observable proxy, since hidden states are informative but may be too expensive or invasive for the final indicator.

### Next step
- Start Phase 3 with a new full-profile script that logs aligned per-position pre-final features across the entire settled span, including mismatch positions, then validate whether the strongest Phase 2 patterns hold at scale.

## SESSION HANDOFF — 2026-04-05 19:20:53 KST
### Current phase and sub-step
Phase 2 complete; synthesis written, ready to enter Phase 3 full-profile data collection.

### State of work
- Real 32k runs for Phase 2b, 2c, and 2d are complete.
- `phase2_summary.md` has been written.
- No production code paths were modified; only research artifacts and markdown summaries were updated.

### Critical context the next session MUST know
- Use `conda run -n retroinfer python ...`; base `python` still lacks `torch`.
- Phase 2b confirmed a logging-alignment caveat: final settled blocks can contain mismatch positions with no same-cycle early/draft payload in the CSV. This is now a design requirement for the Phase 3 profiler.
- Hidden-state drift was the clearest separation signal in Phase 2: rejected-block cosine/L2 drift was much larger than accepted-block drift, and the left neighbor of the rejection could already be severely distorted.
- Small budget perturbations around budget2 did not fix the rejected positions; all observed rejected positions stayed wrong at 0.08, 0.10, and 0.12 retrieval budgets.

### Exact next action
- Create a new Phase 3 profiling script under `tests/RetrievalAttention_new/research/` that logs aligned per-position draft and early-verify features across the entire final-settled span, then run it on the same `Meta-Llama-3.1-8B`, `prefix_len=32768` configuration.

### Files to read first
- `tests/RetrievalAttention_new/research/research_log.md`
- `tests/RetrievalAttention_new/research/phase2_summary.md`
- `tests/RetrievalAttention_new/research/data/phase2b_logit_probe_Meta-Llama-3.1-8B_32768.csv`
- `tests/RetrievalAttention_new/research/data/phase2c_hidden_state_probe_Meta-Llama-3.1-8B_32768.csv`
- `tests/RetrievalAttention_new/research/data/phase2d_budget_sensitivity_Meta-Llama-3.1-8B_32768.csv`

## [Phase 3.1] Aligned full-profile implementation and first real run — 2026-04-05 21:09:11 KST
### What was done
- Added a new research profiler:
- `tests/RetrievalAttention_new/research/phase3_full_profile.py`
- Added the Phase 3 command entry to:
- `tests/RetrievalAttention_new/research/scripts.md`
- Smoke-validated the new profiler in the required runtime environment:
- `CUDA_VISIBLE_DEVICES=1 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_full_profile.py --dataset pg19 --model_name Meta-Llama-3.1-8B --prefix_len 2048 --num_max_token 8 --num_eval_steps 1 --max_cycles 1 --gamma1 2 --gamma2 4 --budget1 0.02 --budget2 0.10 --budget2_high 0.20 --enable_dynamic_budget --T_low 0.05 --T_high 0.2 --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_smoke.csv`
- Ran the real Phase 3 collection on the target configuration:
- `CUDA_VISIBLE_DEVICES=1 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_full_profile.py --dataset pg19 --model_name Meta-Llama-3.1-8B --prefix_len 32768 --num_max_token 100 --num_eval_steps 1 --max_cycles 10 --gamma1 6 --gamma2 32 --budget1 0.02 --budget2 0.10 --budget2_high 0.20 --enable_dynamic_budget --T_low 0.05 --T_high 0.2 --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768.csv`
- Analyzed the generated CSV for coverage and first candidate separations.
- Wrote the first Phase 3 summary:
- `tests/RetrievalAttention_new/research/phase3_summary.md`

### Key findings
- Phase 3 solved the central Phase 2 alignment problem:
- output artifact `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768.csv`
- size 183 KB, 135 rows across 5 settled cycles.
- 130/130 non-bonus settled positions had `pre_final_features_available=1`.
- the only rows without pre-final features were the 5 final authoritative bonus rows, which is expected because those tokens are created only at final verify.
- The actual mismatch rows are now directly observable with aligned pre-final features:
- cycle 2 mismatch at position 8 came from an `early_bonus` under `high` mode.
- cycle 4 mismatch at position 16 came from an `accepted_prefix` under `normal` mode.
- Both mismatch rows had notably weak pre-final certainty:
- early margins `0.0593` and `0.0399`
- early entropies `3.8514` and `3.7592`
- KL(final||early) `0.0118` and `0.0513`
- top-10 early/final overlap was still high at `9/10` for both, so the rejection signature here is not “completely different candidate set.” It is a fragile, near-tie local ranking problem inside an otherwise overlapping shortlist.
- Relative to accepted non-bonus positions, rejected exact mismatch rows looked much softer:
- accepted non-bonus positions: mean early margin `0.4173`, mean early entropy `2.0398`, mean KL(final||early) `0.0064`, mean top-10 overlap `9.56`
- exact mismatch rows: mean early margin `0.0496`, mean early entropy `3.8053`, mean KL(final||early) `0.0315`, mean top-10 overlap `9.0`
- The immediate left neighbors of the mismatches did not collapse in the same way:
- rel -1 rows had mean early margin `0.7650`, mean early entropy `1.0984`, mean KL(final||early) `0.0098`
- this suggests the decisive signal is sharply localized at the mismatch position itself in logit space, even though Phase 2 hidden-state drift had already appeared slightly earlier.
- Source type alone is not enough:
- early-bonus positions are generally softer than accepted-prefix positions, but most still settle fine.
- accepted-prefix positions: mean early margin `0.4268`, mean early entropy `2.2311`
- early-bonus positions: mean early margin `0.2017`, mean early entropy `2.8883`
- exact mismatch rows were softer than both groups.
- A simple candidate indicator family now looks plausible:
- `early_margin < 0.1` and `early_entropy > 3.5` captured both observed mismatch rows.
- `early_margin < 0.1` and `KL(final||early) > 0.01` also captured both mismatch rows and was rarer on accepted positions (`4/72` accepted non-bonus rows).
- Important constraint clarification:
- the final goal is a pre-final indicator, so any rule that uses final-verify outputs directly is not a valid online predictor.
- Therefore `KL(final||early)` and final-overlap/rank features remain useful for offline analysis only.
- The current best valid online candidate is the early-only fragility rule:
- `early_margin < 0.1` and `early_entropy > 3.5`

### Open questions
- This first Phase 3 run still has only 2 exact mismatch rows, so the candidate thresholds are suggestive rather than stable.
- Need more Phase 3 data to test whether low-margin/high-entropy mismatch signatures keep holding across more prompts and cycles.
- Need to compare whether the strongest practical predictor comes from:
- early-only observables (`margin`, `entropy`, top-2 gap),
- cross-stage observables that need a more expensive alignment replay (`KL`, overlap),
- or a hybrid gated rule.
- Need to focus the next search on pre-final-only features. Cross-stage-to-final signals are allowed for offline diagnosis but must not be treated as deployable indicators.

### Next step
- Stay in Phase 3 and scale the aligned full-profile collection to additional evaluation steps / cycles, then test simple candidate rejection indicators derived from the new aligned ledger.

## SESSION HANDOFF — 2026-04-05 21:09:11 KST
### Current phase and sub-step
Phase 3 started successfully; aligned full-profile profiler implemented, real 32k first run complete, first findings written.

### State of work
- New script exists:
- `tests/RetrievalAttention_new/research/phase3_full_profile.py`
- New artifact exists:
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768.csv`
- `scripts.md` has been updated.
- `phase3_summary.md` has been written.
- No production code paths were modified; only research artifacts and markdown notes changed.

### Critical context the next session MUST know
- Use `conda run -n retroinfer python ...`; base `python` still does not have `torch`.
- Phase 3 fixed the Phase 2b alignment caveat: every non-bonus settled position now has attached pre-final draft/early features in the CSV.
- In this first aligned run, the true mismatch rows were not characterized by zero overlap or huge rank collapse. They were characterized by very low early margin, high early entropy, and moderate KL despite still sharing `9/10` top-10 tokens with final verify.
- One mismatch came from an `early_bonus` in `high` mode and the other from an `accepted_prefix` in `normal` mode, so rejection is not confined to one source kind.

### Exact next action
- Extend Phase 3 collection beyond the first eval step and test only pre-final candidate indicators on the aligned CSVs, starting with low-margin/high-entropy and draft-vs-early disagreement rules.

### Files to read first
- `tests/RetrievalAttention_new/research/research_log.md`
- `tests/RetrievalAttention_new/research/phase3_summary.md`
- `tests/RetrievalAttention_new/research/phase2_summary.md`
- `tests/RetrievalAttention_new/research/phase3_full_profile.py`
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768.csv`

## [Phase 3.2] GPU0 expansion and pre-final rule scan — 2026-04-06 16:46:58 KST
### What was done
- Switched subsequent experiment runs to `CUDA_VISIBLE_DEVICES=0` per updated runtime guidance.
- Added a new Phase 3 analysis helper:
- `tests/RetrievalAttention_new/research/phase3_pre_final_indicator_scan.py`
- Added the new script to:
- `tests/RetrievalAttention_new/research/scripts.md`
- Ran an intermediate expanded aligned profile on GPU0:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_full_profile.py --dataset pg19 --model_name Meta-Llama-3.1-8B --prefix_len 32768 --num_max_token 100 --num_eval_steps 5 --max_cycles 10 --gamma1 6 --gamma2 32 --budget1 0.02 --budget2 0.10 --budget2_high 0.20 --enable_dynamic_budget --T_low 0.05 --T_high 0.2 --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps5.csv`
- Scanned pre-final block-level rules on that intermediate artifact.
- Ran a larger aligned profile on GPU0:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_full_profile.py --dataset pg19 --model_name Meta-Llama-3.1-8B --prefix_len 32768 --num_max_token 100 --num_eval_steps 5 --max_cycles 25 --gamma1 6 --gamma2 32 --budget1 0.02 --budget2 0.10 --budget2_high 0.20 --enable_dynamic_budget --T_low 0.05 --T_high 0.2 --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps5_cycles25.csv`
- Re-ran the pre-final rule scan on the larger artifact:
- `python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_pre_final_indicator_scan.py --input_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps5_cycles25.csv --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_indicator_scan_Meta-Llama-3.1-8B_32768_steps5_cycles25.csv`
- Updated:
- `tests/RetrievalAttention_new/research/phase3_summary.md`

### Key findings
- Intermediate GPU0 artifact:
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps5.csv`
- 301 rows, 10 settled blocks, 5 rejected blocks.
- On that smaller expansion, a draft-vs-early disagreement rule already beat the original entropy-only gate:
- `early_margin < 0.10 and top10_overlap_draft_early <= 8`
- precision `0.800`, recall `0.800` (`4 TP`, `1 FP`, `1 FN`)
- Larger GPU0 artifact:
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps5_cycles25.csv`
- 664 rows across 20 settled blocks from steps `0-4`, with 8 rejected blocks and 8 directly observed mismatch rows.
- The mismatch rows in the larger run remained extremely fragile at early verify:
- early margins: `0.0464`, `0.0082`, `0.0394`, `0.0049`, `0.0183`, `0.0000`, `0.0000`, `0.0044`
- early entropies: `3.8741`, `4.2507`, `3.7538`, `4.2654`, `5.4510`, `3.8528`, `6.0473`, `5.0965`
- Source mix of mismatch rows:
- 5 were `early_bonus`
- 3 were `accepted_prefix`
- Therefore source kind helps, but is not sufficient by itself.
- The original early-only gate degraded materially on the larger sample:
- `early_margin < 0.1 and early_entropy > 3.5`
- precision `0.500`, recall `1.000`, `8 TP`, `8 FP`, `0 FN`
- The best current valid pre-final rule from the larger scan is:
- `early_margin < 0.03 and KL(early_verify || draft) > 0.01`
- precision `0.700`, recall `0.875`, F1 `0.778`, trigger rate `0.500`, `7 TP`, `3 FP`, `1 FN`
- `KL(early_verify || draft) > 0.02` gave the same result in this sample.
- This is still a valid online rule because it uses only pre-final draft and early-verify distributions.
- The best overlap-style rule on the larger sample was weaker:
- `early_margin < 0.10 and top10_overlap_draft_early <= 8`
- precision `0.625`, recall `0.625`
- Updated mechanistic interpretation:
- pure early uncertainty is too broad and fires on many accepted blocks,
- but combining low early margin with strong draft-vs-early divergence is more selective and remains high-recall.
- The strongest online signal so far is therefore "fragile early verify plus substantial disagreement with the draft distribution."

### Open questions
- Even the best current valid pre-final rule is still below the target:
- precision `0.700` < required `0.900`
- Need more Phase 3 data and more rule families, especially:
- position-aware or source-aware piecewise rules,
- worst-position vs count-based block rules,
- two-stage filters that combine cheap uncertainty with draft-vs-early drift.
- Need to understand the one missed rejected block under the current best rule and the three accepted blocks it still false-alarmed on.

### Next step
- Continue Phase 3 on GPU0 and expand the rule scan to:
- more thresholds around low-margin + draft-vs-early KL,
- position-aware rules for `early_bonus` vs `accepted_prefix`,
- and block-level aggregations beyond "any position triggers."

## SESSION HANDOFF — 2026-04-06 16:46:58 KST
### Current phase and sub-step
Phase 3 ongoing; GPU0 expanded aligned collection and first reproducible pre-final rule scan are complete.

### State of work
- New analysis script exists:
- `tests/RetrievalAttention_new/research/phase3_pre_final_indicator_scan.py`
- New GPU0 artifacts exist:
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps5.csv`
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps5_cycles25.csv`
- `tests/RetrievalAttention_new/research/data/phase3_indicator_scan_Meta-Llama-3.1-8B_32768_steps5_cycles25.csv`
- `phase3_summary.md` has been updated with the larger-run findings.

### Critical context the next session MUST know
- Use `CUDA_VISIBLE_DEVICES=0` for subsequent runs.
- The current best valid online candidate is no longer the entropy-only rule.
- Best current pre-final rule on the expanded GPU0 sample:
- `early_margin < 0.03 and KL(early_verify || draft) > 0.01`
- Metrics on 20 settled blocks / 8 rejected blocks:
- precision `0.700`, recall `0.875`, `7 TP`, `3 FP`, `1 FN`
- The earlier entropy-only gate
- `early_margin < 0.1 and early_entropy > 3.5`
- dropped to precision `0.500` on the larger sample.
- The best signal so far is a hybrid of low early certainty plus strong draft-vs-early drift, not early uncertainty alone.

### Exact next action
- Extend the scan with more pre-final-only rule families, especially position-aware and source-aware block rules, while keeping all runs on `CUDA_VISIBLE_DEVICES=0`.

### Files to read first
- `tests/RetrievalAttention_new/research/research_log.md`
- `tests/RetrievalAttention_new/research/phase3_summary.md`
- `tests/RetrievalAttention_new/research/phase3_pre_final_indicator_scan.py`
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps5_cycles25.csv`
- `tests/RetrievalAttention_new/research/data/phase3_indicator_scan_Meta-Llama-3.1-8B_32768_steps5_cycles25.csv`

## [Phase 3.3] Corrected steps20 per-step collection and larger scan — 2026-04-06 18:34:47 KST
### What was done
- Ran the requested larger Phase 3 profile on GPU0 with:
- `num_eval_steps=20`
- `num_max_token=100`
- `max_cycles=25`
- Initial command:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_full_profile.py --dataset pg19 --model_name Meta-Llama-3.1-8B --prefix_len 32768 --num_max_token 100 --num_eval_steps 20 --max_cycles 25 --gamma1 6 --gamma2 32 --budget1 0.02 --budget2 0.10 --budget2_high 0.20 --enable_dynamic_budget --T_low 0.05 --T_high 0.2 --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25.csv`
- Inspected the resulting artifact and found a real research-profiler bug:
- `max_cycles` was being applied globally across the whole run instead of per eval step.
- This caused the first steps20 artifact to stop after 25 total settled cycles, covering only steps `0-6`.
- Fixed the bug in:
- `tests/RetrievalAttention_new/research/phase3_full_profile.py`
- Specifically, `cycle_count` now resets inside each eval step loop.
- Re-validated syntax with:
- `conda run -n retroinfer python -m py_compile /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_full_profile.py`
- Re-ran the corrected steps20 collection on GPU0 with a new output path:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_full_profile.py --dataset pg19 --model_name Meta-Llama-3.1-8B --prefix_len 32768 --num_max_token 100 --num_eval_steps 20 --max_cycles 25 --gamma1 6 --gamma2 32 --budget1 0.02 --budget2 0.10 --budget2_high 0.20 --enable_dynamic_budget --T_low 0.05 --T_high 0.2 --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- Re-ran the block-level pre-final scan on the corrected artifact:
- `python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_pre_final_indicator_scan.py --input_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_indicator_scan_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- Ran an extra precision-oriented threshold sweep around low-margin + draft-vs-early-KL and several piecewise variants.
- Updated:
- `tests/RetrievalAttention_new/research/phase3_summary.md`

### Key findings
- The first steps20 artifact should be treated as partial / pre-fix only:
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25.csv`
- 804 rows, 25 settled blocks, only steps `0-6` covered.
- Corrected per-step artifact:
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- 2715 rows across 88 settled blocks, with 36 rejected blocks and 36 directly observed mismatch rows.
- Corrected run covers all requested steps:
- steps `0-19`
- A tokenizer warning appeared for one overlength sample:
- `133585 > 131072`
- but the corrected run still completed and logged all 20 eval steps.
- Mismatch composition changed materially at scale:
- 21 mismatch rows were `accepted_prefix`
- 15 mismatch rows were `early_bonus`
- 31/36 mismatch rows used `high` verify mode, 5/36 used `normal`
- Therefore the larger sample confirms:
- accepted-prefix failures are at least as important as bonus-token failures,
- and rejections are concentrated heavily in `high` mode.
- On the corrected steps20 sample, the old best rule from the smaller run degraded:
- `early_margin < 0.03 and KL(early_verify || draft) > 0.01`
- precision `0.583`, recall `0.639`
- The best precision among scanned valid pre-final rules at recall >= 0.5 became:
- `early_margin < 0.03 and KL(early_verify || draft) > 0.02`
- precision `0.783`, recall `0.500`, `18 TP`, `5 FP`, `18 FN`
- The strongest balanced KL-based rule in the scan became:
- `early_margin < 0.10 and KL(early_verify || draft) > 0.02`
- precision `0.667`, recall `0.833`, F1 `0.741`, `30 TP`, `15 FP`, `6 FN`
- The best overlap-based rule on the corrected steps20 sample was:
- `early_margin < 0.10 and top10_overlap_draft_early <= 8`
- precision `0.700`, recall `0.875`, F1 `0.778`, `19 TP`, `10 FP`, `17 FN`
- This means the best F1 on this scan came from the overlap-collapse rule, while the best precision at usable recall came from the stricter KL-based rule.
- Mechanistic update from the larger sample:
- pure early-only fragility remains too broad,
- draft-vs-early drift remains consistently useful,
- and there may be two partial rejection subfamilies:
- overlap-collapse cases,
- and low-margin / moderate-drift cases that still keep much of the same shortlist.

### Open questions
- No scanned rule reaches the target:
- precision >= 0.900 with recall >= 0.500
- Need better block-level logic than simple any-position triggers.
- The accepted-prefix mismatch rows show heterogeneous draft-vs-early signatures:
- some have high KL,
- some have low KL but low margin and only mild overlap degradation.
- Need to inspect the 5 false positives under the `m<0.03 & kl>0.02` rule and the 18 false negatives it misses.

### Next step
- Continue Phase 3 with source-aware and count-based block rules on the corrected steps20 artifact, e.g.:
- separate thresholds for `early_bonus` vs `accepted_prefix`
- require 2 weak positions in a block instead of 1
- combine overlap collapse with low-margin drift in a piecewise rule

## SESSION HANDOFF — 2026-04-06 18:34:47 KST
### Current phase and sub-step
Phase 3 ongoing; corrected steps20 per-step artifact and larger pre-final scan are complete.

### State of work
- `phase3_full_profile.py` has been fixed so `max_cycles` resets per eval step.
- New corrected artifact exists:
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- New corrected scan exists:
- `tests/RetrievalAttention_new/research/data/phase3_indicator_scan_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- `phase3_summary.md` has been updated with the corrected steps20 findings.

### Critical context the next session MUST know
- Use `CUDA_VISIBLE_DEVICES=0`.
- Treat `phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25.csv` as a pre-fix partial artifact only.
- Use the corrected per-step artifact for further Phase 3 analysis:
- `phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- Corrected steps20 sample stats:
- 88 settled blocks
- 36 rejected blocks
- 36 directly observed mismatch rows
- Current best precision at recall >= 0.5 in the scanned rule family:
- `early_margin < 0.03 and KL(early_verify || draft) > 0.02`
- precision `0.783`, recall `0.500`
- Current best balanced rule in the scan:
- `early_margin < 0.10 and top10_overlap_draft_early <= 8`
- precision `0.700`, recall `0.875`
- or
- `early_margin < 0.10 and KL(early_verify || draft) > 0.02`
- precision `0.667`, recall `0.833`
- No rule yet hits the target `>=90%` precision with `>=50%` recall.

### Exact next action
- Design and evaluate source-aware / count-based block rules on the corrected steps20 artifact, using only pre-final features.

### Files to read first
- `tests/RetrievalAttention_new/research/research_log.md`
- `tests/RetrievalAttention_new/research/phase3_summary.md`
- `tests/RetrievalAttention_new/research/phase3_full_profile.py`
- `tests/RetrievalAttention_new/research/phase3_pre_final_indicator_scan.py`
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- `tests/RetrievalAttention_new/research/data/phase3_indicator_scan_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`

## [Phase 3.4] Source-aware multi-hit pre-final scan on corrected steps20 artifact — 2026-04-06 20:08:43 KST
### What was done
- Added a new research-only analysis script:
- `tests/RetrievalAttention_new/research/phase3_advanced_pre_final_scan.py`
- Updated:
- `tests/RetrievalAttention_new/research/scripts.md`
- Syntax-checked the new script with:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python -m py_compile /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_advanced_pre_final_scan.py`
- Ran the new advanced Phase 3 rule search on the corrected steps20 artifact with:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_advanced_pre_final_scan.py --input_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv --details_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep_details.csv --top_k 200`
- The new scan explicitly searched:
- source-aware rules (`accepted_prefix` vs `early_bonus`)
- count-based multi-hit rules instead of simple any-position triggers
- position-aware bonus rules
- piecewise hybrids that combine low early margin with draft-vs-early KL / overlap on accepted-prefix rows

### Key findings
- The corrected steps20 artifact now has a clearly better valid pre-final rule than the prior best `early_margin < 0.03 and KL(early_verify || draft) > 0.02`.
- New best valid pre-final block rule on the corrected steps20 artifact:
- trigger if either:
- at least `2` `accepted_prefix` rows satisfy `early_margin < 0.05` and (`KL(early_verify || draft) > 0.03` or `top10_overlap_draft_early <= 7`)
- or at least `2` `early_bonus` rows at position `>= 4` satisfy `early_margin < 0.03` and `early_entropy > 3.8`
- Metrics of this source-aware multi-hit rule:
- precision `0.909`
- recall `0.556`
- F1 `0.690`
- `20 TP`, `2 FP`, `16 FN`
- trigger rate `0.250`
- This is a valid online / deployable Phase 3 indicator because it uses only pre-final features:
- `source_kind`
- position
- early margin / entropy
- and draft-vs-early KL / overlap for accepted-prefix rows
- It does **not** use final-verify outputs.
- Therefore this corrected steps20 scan now reaches the stated target:
- precision `>= 0.900`
- recall `>= 0.500`
- Mechanistic interpretation of why this beats the older any-position rules:
- many accepted blocks contain one weak row, but far fewer contain two weak rows from the same source family
- accepted-prefix failures are best captured by repeated low-margin draft-vs-early drift or shortlist collapse
- early-bonus failures are best captured by repeated late-position low-margin high-entropy instability
- The two source-specific subrules were each high-precision but too low-recall on their own:
- accepted-prefix subrule alone:
- precision `0.929`, recall `0.361`, `13 TP`, `1 FP`
- early-bonus subrule alone:
- precision `0.917`, recall `0.306`, `11 TP`, `1 FP`
- Their OR-combination is what crosses the target by covering complementary rejection subfamilies.
- The winning rule still misses 16 rejected blocks, mostly blocks with only a single weak hit or no repeated weak pre-final signatures under this threshold family.

### Open questions
- Need to understand whether the new `0.909 / 0.556` rule is stable under additional data, or is partly tuned to the corrected steps20 sample.
- The remaining 16 false negatives suggest a second rejection family still exists:
- blocks with only one weak accepted-prefix signal,
- or blocks with a single weak bonus signal but not repeated instability.
- The next robustness step should be validation on a larger corrected artifact or on another contiguous shard, rather than more threshold tuning on the same sample.

### Next step
- Treat the current source-aware multi-hit rule as the new best valid Phase 3 candidate and move next to robustness testing / holdout confirmation.

## SESSION HANDOFF — 2026-04-06 20:08:43 KST
### Current phase and sub-step
Phase 3 remains active, but the corrected steps20 artifact now has a valid pre-final rule that clears the target on-sample.

### State of work
- New script exists:
- `tests/RetrievalAttention_new/research/phase3_advanced_pre_final_scan.py`
- New artifacts exist:
- `tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- `tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep_details.csv`
- `scripts.md` and `phase3_summary.md` have been updated to include this scan.

### Critical context the next session MUST know
- Use `CUDA_VISIBLE_DEVICES=0`.
- Use `conda run -n retroinfer python ...` for all runs.
- Continue to treat:
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25.csv`
- as a pre-fix partial artifact only.
- Use the corrected artifact:
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- The new best valid pre-final rule on the corrected steps20 artifact is:
- trigger if either:
- at least `2` `accepted_prefix` rows satisfy `early_margin < 0.05` and (`KL(early_verify || draft) > 0.03` or `top10_overlap_draft_early <= 7`)
- or at least `2` `early_bonus` rows at position `>= 4` satisfy `early_margin < 0.03` and `early_entropy > 3.8`
- Metrics:
- precision `0.909`
- recall `0.556`
- `20 TP`, `2 FP`, `16 FN`
- This beats the previous best:
- `early_margin < 0.03 and KL(early_verify || draft) > 0.02`
- which had precision `0.783`, recall `0.500`

### Exact next action
- Validate the new source-aware multi-hit rule on more corrected data or a holdout shard before tightening thresholds further.

### Files to read first
- `tests/RetrievalAttention_new/research/research_log.md`
- `tests/RetrievalAttention_new/research/phase3_summary.md`
- `tests/RetrievalAttention_new/research/phase3_advanced_pre_final_scan.py`
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- `tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- `tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep_details.csv`

## [Phase 3.5] Corrected steps50 robustness validation — 2026-04-07 10:03:07 KST
### What was done
- Ran the corrected larger Phase 3 profile on GPU0 with:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_full_profile.py --dataset pg19 --model_name Meta-Llama-3.1-8B --prefix_len 32768 --num_max_token 100 --num_eval_steps 50 --max_cycles 25 --gamma1 6 --gamma2 32 --budget1 0.02 --budget2 0.10 --budget2_high 0.20 --enable_dynamic_budget --T_low 0.05 --T_high 0.2 --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv`
- Re-ran the original Phase 3 simple rule scan on the new artifact:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_pre_final_indicator_scan.py --input_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_indicator_scan_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv`
- Re-ran the advanced source-aware multi-hit scan on the new artifact:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_advanced_pre_final_scan.py --input_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv --output_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv --details_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep_details.csv --top_k 200`

### Key findings
- Corrected `steps50` artifact:
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv`
- contains:
- 7163 rows
- 231 settled blocks
- 109 rejected blocks
- 109 directly observed mismatch rows
- The old simple `steps20` best rule degraded on the larger sample:
- `early_margin < 0.03 and KL(early_verify || draft) > 0.02`
- precision `0.714`
- recall `0.550`
- `60 TP`, `24 FP`, `49 FN`
- The old source-aware `steps20` winning rule also degraded materially:
- precision `0.824`
- recall `0.385`
- `42 TP`, `9 FP`, `67 FN`
- Therefore the `>=0.900` precision / `>=0.500` recall result from `steps20` did **not** hold under the larger `steps50` validation sample.
- On `steps50`, the best simple 2-metric rule from the scanned family became:
- `early_margin < 0.03 and KL(early_verify || draft) > 0.01`
- precision `0.688`
- recall `0.688`
- F1 `0.688`
- `75 TP`, `34 FP`, `34 FN`
- On `steps50`, the best advanced scanned source-aware rule became:
- trigger if either:
- at least `2` `accepted_prefix` rows satisfy `early_margin < 0.05` and (`KL(early_verify || draft) > 0.03` or `top10_overlap_draft_early <= 7`)
- or at least `2` `early_bonus` rows satisfy `early_margin < 0.05` and `early_entropy > 3.5`
- Metrics:
- precision `0.824`
- recall `0.514`
- F1 `0.633`
- `56 TP`, `12 FP`, `53 FN`
- This new advanced `steps50` winner still beats the old simple `steps20` best rule on precision at usable recall, but it no longer reaches the Phase 3 target.
- Mechanistic update from the robustness run:
- the source-aware multi-hit idea remains useful,
- but the exact `steps20` thresholds were too optimistic / sample-tuned,
- and broader `early_bonus` instability appears necessary once the sample grows.

### Open questions
- Need to understand whether precision can be recovered above `0.90` with a richer but still deployable block rule, or whether the current pre-final feature family has now saturated below target.
- The larger `steps50` sample suggests the main remaining gap is not discovering the right rule skeleton, but stabilizing thresholds across more heterogeneous rejection cases.
- It may now be necessary to try:
- normalized / rank-based versions of KL or overlap,
- per-block summary statistics beyond counts,
- or separate validation/selection splits instead of tuning on one artifact.

### Next step
- Treat `steps50` as the first real robustness check and move next to either:
- a train/validation split inside the current artifact family, or
- a new rule family with stronger calibration / block summary features.

## SESSION HANDOFF — 2026-04-07 10:03:07 KST
### Current phase and sub-step
Phase 3 robustness validation on corrected `steps50` is complete.

### State of work
- New corrected validation artifact exists:
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv`
- New simple scan exists:
- `tests/RetrievalAttention_new/research/data/phase3_indicator_scan_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv`
- New advanced scan exists:
- `tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv`
- `tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep_details.csv`

### Critical context the next session MUST know
- Use `CUDA_VISIBLE_DEVICES=0`.
- Use `conda run -n retroinfer python ...` for all runs.
- The corrected `steps20` winner did not survive the larger `steps50` validation.
- Old simple rule on `steps50`:
- `early_margin < 0.03 and KL(early_verify || draft) > 0.02`
- precision `0.714`, recall `0.550`
- Old advanced `steps20` winner on `steps50`:
- precision `0.824`, recall `0.385`
- Best simple scanned rule on `steps50`:
- `early_margin < 0.03 and KL(early_verify || draft) > 0.01`
- precision `0.688`, recall `0.688`
- Best advanced scanned rule on `steps50`:
- precision `0.824`, recall `0.514`
- No scanned rule on `steps50` reaches the target `>=0.900` precision with `>=0.500` recall.

### Exact next action
- Continue Phase 3 with robustness-aware rule design rather than trusting the `steps20` thresholds.

### Files to read first
- `tests/RetrievalAttention_new/research/research_log.md`
- `tests/RetrievalAttention_new/research/phase3_summary.md`
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv`
- `tests/RetrievalAttention_new/research/data/phase3_indicator_scan_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv`
- `tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv`

## [Phase 3.6] Mechanistic interaction analysis for accepted-prefix margin+KL — 2026-04-07 10:29:46 KST
### What was done
- Added a new research-only analysis script:
- `tests/RetrievalAttention_new/research/phase3_interaction_analysis.py`
- Updated:
- `tests/RetrievalAttention_new/research/scripts.md`
- Syntax-checked the new script with:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python -m py_compile /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_interaction_analysis.py`
- Attempted to run the analysis in `retroinfer`, but that environment does not currently include `matplotlib`, so the figure-generation run was executed with system `python` instead:
- `python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_interaction_analysis.py --steps20_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv --steps50_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv --output_dir /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data`
- Generated new analysis artifacts:
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_conditional_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_block_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_mechanism_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_heatmaps.png`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_conditional_lines.png`

### Key findings
- The interaction analysis focused on `accepted_prefix` rows because that is the source family where `KL(early_verify || draft)` is defined and was already implicated by the earlier scans.
- The main mechanistic result is that `KL` is weak as a standalone signal because much of its mass lies in high-margin regions where early verify is still locally stable.
- On accepted-prefix rows in corrected `steps50`:
- overall token reject rate was only `1.11%` (`67/6054`)
- `high_kl_without_low_margin`: `10/526`, reject rate `1.90%`
- `moderate_kl_without_low_margin`: `9/1155`, reject rate `0.78%`
- `low_margin_only`: `13/99`, reject rate `13.13%`
- `low_margin_plus_moderate_kl`: `14/131`, reject rate `10.69%`
- `low_margin_plus_high_kl`: `16/79`, reject rate `20.25%`
- So the dangerous accepted-prefix region is not “high KL everywhere”; it is the subset where KL occurs together with low early margin.
- The new conditional reject-rate table and line plot make the asymmetry explicit. Inside the `high KL` slice on `steps50`:
- `low(<0.03)` margin rows reject at `20.25%` (`16/79`)
- `0.03-0.10` margin rows reject at `4.43%` (`7/158`)
- `>=0.10` margin rows reject at only `0.82%` (`3/368`)
- This explains why the earlier KL-only histogram looked weak: KL alone pools together a small dangerous low-margin subset with a much larger harmless high-margin subset.
- The 2D heatmap adds the same evidence visually:
- the highest reject-rate bins sit in the low-margin corner with moderate-to-high KL,
- while many dense high-margin bins remain close to zero reject rate even when KL is nontrivial.
- The moderate-KL regime is not just noise or an extreme-tail artifact.
- In corrected `steps50`, among blocks that already contain a low-margin accepted-prefix row:
- baseline reject rate of that subset is `65.4%` (`89/136`)
- requiring at least one `low_margin + moderate_kl` hit yields precision `0.688` with `55 TP`, `25 FP`, recall `0.505`
- requiring at least one `low_margin + high_kl` hit yields cleaner precision `0.721` with `44 TP`, `17 FP`, recall `0.404`
- So moderate KL does carry useful signal once margin says the row is already brittle; it is not only the extreme high-KL tail that matters.
- Repeated joint hits isolate a purer but smaller rejection family.
- In corrected `steps50`, at least `2` accepted-prefix rows with `low_margin + high_kl` gives:
- precision `0.938`
- recall `0.138`
- `15 TP`, `1 FP`
- This matches the earlier source-aware scan intuition: repeated fragile-and-drifting hits are very pure, but not broad enough to explain all rejections alone.
- The same qualitative interaction pattern is present on corrected `steps20`, but the larger `steps50` sample confirms that the mechanism is real even though the exact `steps20` thresholds were too optimistic.

### Mechanistic interpretation
- `early_margin` behaves like a local fragility signal.
- `KL(early_verify || draft)` behaves like a draft-vs-early drift signal.
- Drift without fragility is often harmless.
- Fragility without drift is broader and noisier.
- The accepted-prefix rejection family that the combined rule helps recover is the intersection:
- rows that are already locally brittle and whose early distribution has already moved away from the draft before final verify runs.
- This is why `early_margin + KL` helps more than either metric alone even though KL by itself looks weak in the earlier 1D histogram.

### Open questions
- The accepted-prefix interaction story is now much clearer, but it still does not by itself deliver a robust `>=0.900` precision / `>=0.500` recall rule on `steps50`.
- Need to determine whether the next improvement should come from:
- better block summarization of the accepted-prefix interaction,
- normalized / rank-based KL variants,
- or combining this accepted-prefix mechanism with a separately stabilized early-bonus mechanism.
- It may also be worth adding confidence intervals / bootstrap uncertainty to the interaction tables so the paper-ready claims can be stated more defensibly.

### Next step
- Use the new interaction evidence to design robustness-aware block summaries rather than more raw threshold scans, especially:
- count or fraction summaries of low-margin + moderate/high-KL accepted-prefix hits
- and a cleaner decomposition of accepted-prefix versus early-bonus rejection families on `steps50`.

## SESSION HANDOFF — 2026-04-07 10:29:46 KST
### Current phase and sub-step
Phase 3 interaction analysis is complete for the accepted-prefix margin+KL mechanism.

### State of work
- New script exists:
- `tests/RetrievalAttention_new/research/phase3_interaction_analysis.py`
- New artifacts exist:
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_conditional_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_block_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_mechanism_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_heatmaps.png`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_conditional_lines.png`
- `tests/RetrievalAttention_new/research/phase3_summary.md` and `tests/RetrievalAttention_new/research/scripts.md` have been updated.

### Critical context the next session MUST know
- Use `CUDA_VISIBLE_DEVICES=0`.
- Use `conda run -n retroinfer python ...` for profiling / scan runs.
- The new interaction analysis script can be syntax-checked in `retroinfer`, but the current `retroinfer` environment does not have `matplotlib`, so the figure-generation run was done with system `python`.
- The accepted-prefix mechanism is now much clearer:
- KL alone looks weak because many moderate/high-KL rows sit in safe high-margin regions.
- Low margin is the necessary “fragility” context.
- Moderate KL under low margin is already useful; the signal is not only the extreme high-KL tail.
- Repeated low-margin + high-KL hits are very pure but too low-recall on their own.
- On corrected `steps50`, especially important reference numbers are:
- token-level:
- `high_kl_without_low_margin`: `10/526`, `1.90%`
- `moderate_kl_without_low_margin`: `9/1155`, `0.78%`
- `low_margin_only`: `13/99`, `13.13%`
- `low_margin_plus_moderate_kl`: `14/131`, `10.69%`
- `low_margin_plus_high_kl`: `16/79`, `20.25%`
- block-level:
- `any low-margin accepted-prefix row`: precision `0.654`, recall `0.817`
- `any low-margin + moderate-KL accepted-prefix row`: precision `0.688`, recall `0.505`
- `any low-margin + high-KL accepted-prefix row`: precision `0.721`, recall `0.404`
- `>=2 low-margin + high-KL accepted-prefix rows`: precision `0.938`, recall `0.138`

### Exact next action
- Build the next robustness-aware rule family around block summaries of the accepted-prefix interaction and explicitly test how it combines with the early-bonus family on `steps50`.

### Files to read first
- `tests/RetrievalAttention_new/research/research_log.md`
- `tests/RetrievalAttention_new/research/phase3_summary.md`
- `tests/RetrievalAttention_new/research/phase3_interaction_analysis.py`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_conditional_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_block_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_mechanism_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_heatmaps.png`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_conditional_lines.png`

## [Phase 4] Robustness-aware indicator design around accepted-prefix margin+KL — 2026-04-07 11:08:29 KST
### What was done
- Added a new research-only analysis script:
- `tests/RetrievalAttention_new/research/phase4_indicator_design.py`
- Updated:
- `tests/RetrievalAttention_new/research/scripts.md`
- Added a dedicated summary:
- `tests/RetrievalAttention_new/research/phase4_summary.md`
- Syntax-checked the new script with:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python -m py_compile /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase4_indicator_design.py`
- Ran the Phase 4 cross-dataset analysis with:
- `CUDA_VISIBLE_DEVICES=0 conda run -n retroinfer python /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase4_indicator_design.py --steps20_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv --steps50_csv /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv --output_dir /home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data --output_prefix phase4_indicator`
- Generated new Phase 4 artifacts:
- `tests/RetrievalAttention_new/research/data/phase4_indicator_block_features.csv`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_token_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_rule_metrics.csv`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_shortlist.csv`

### Key findings
- The accepted-prefix mechanism from Phase 3 survives into Phase 4 with the same qualitative structure:
- low margin is the required fragility context,
- moderate KL matters in addition to the high-KL tail,
- and KL without low margin remains mostly too safe to support a clean rule by itself.
- On accepted-prefix rows, Phase 4's token summary reproduced the Phase 3 story on both datasets:
- `steps20`
- `low_margin_only`: `6/39`, `15.38%`
- `low_margin_plus_moderate_kl`: `6/46`, `13.04%`
- `low_margin_plus_high_kl`: `5/21`, `23.81%`
- `moderate_kl_without_low_margin`: `3/439`, `0.68%`
- `high_kl_without_low_margin`: `1/168`, `0.60%`
- `steps50`
- `low_margin_only`: `13/99`, `13.13%`
- `low_margin_plus_moderate_kl`: `14/131`, `10.69%`
- `low_margin_plus_high_kl`: `16/79`, `20.25%`
- `moderate_kl_without_low_margin`: `9/1155`, `0.78%`
- `high_kl_without_low_margin`: `10/526`, `1.90%`
- Accepted-prefix-only block summaries are deployable and reasonably stable, but they appear capped well below the target on corrected `steps50`.
- Best accepted-prefix-only rule with `steps50` recall `>= 0.50`:
- `accepted_high_count|margin<0p05|count>=1`
- `steps20`: precision `0.720`, recall `0.500`, F1 `0.590`
- `steps50`: precision `0.723`, recall `0.550`, F1 `0.625`
- Best more explicitly interaction-style accepted-prefix-only rule:
- `accepted_weighted_score|margin<0p05|score=2*high+mod|>=3`
- `steps20`: precision `0.731`, recall `0.528`, F1 `0.613`
- `steps50`: precision `0.720`, recall `0.541`, F1 `0.618`
- So the accepted-prefix-only family is real, but its `steps50` precision ceiling in the useful recall range is only about `0.72`.
- The best `steps20`-selected accepted-prefix-only candidate did not validate cleanly:
- `accepted_top2_kl|margin<0p05|top2>=0.02`
- `steps20`: precision `0.857`, recall `0.500`
- `steps50`: precision `0.769`, recall `0.459`
- This is another sign that accepted-prefix-only threshold tuning is still vulnerable to sample optimism.
- The best robustness-aware deployable family kept the accepted-prefix interaction as the core precision branch and added an early-bonus branch only because it materially improved corrected `steps50` performance:
- `source_or|accepted_high_count|margin<0p05|or|bonus_ge0_count>=2`
- trigger if either:
- at least `2` accepted-prefix rows satisfy `early_margin < 0.05` and `KL(early_verify || draft) >= 0.03`
- or at least `2` early-bonus rows satisfy `early_margin < 0.05` and `early_entropy > 3.5`
- Metrics:
- `steps20`: precision `0.821`, recall `0.639`, F1 `0.719`, `23 TP`, `5 FP`, `13 FN`
- `steps50`: precision `0.824`, recall `0.514`, F1 `0.633`, `56 TP`, `12 FP`, `53 FN`
- This rule is important because:
- it is explicitly grounded in the accepted-prefix margin+KL mechanism,
- it remains stable in precision across datasets,
- and it improves `steps50` precision by about `+0.10` absolute over the best accepted-prefix-only rule while staying above the `0.50` recall floor.
- If F1 / recall are prioritized instead of precision, the best family becomes:
- `source_or|accepted_weighted_score|margin<0p05|or|bonus_ge0_count>=1`
- `steps20`: precision `0.659`, recall `0.806`, F1 `0.725`
- `steps50`: precision `0.672`, recall `0.789`, F1 `0.726`
- But this family is too low-precision to be the preferred deployable candidate.
- No Phase 4 rule reached the original target on both datasets:
- precision `>= 0.900`
- recall `>= 0.500`
- Therefore the current evidence supports a usable deployable indicator family, but not the original target claim.

### Mechanistic interpretation
- The accepted-prefix branch still behaves like a repeated fragile-and-drifting detector:
- low margin identifies rows where the early distribution is already brittle,
- KL identifies rows where the early distribution has already moved away from the draft,
- and repeated high-KL fragile hits isolate a purer rejection subset than any single accepted-prefix hit.
- The early-bonus branch should now be treated as a complementary robustness component, not as the main explanatory story.
- It earns inclusion only because accepted-prefix-only interaction summaries appear capped around `0.72` precision on corrected `steps50` when recall is kept near `0.5`.

### Open questions
- The Phase 4 candidate family is deployable, but it still falls materially short of the original precision target.
- Need to determine whether the next gain can come from richer accepted-prefix block summaries that keep the same mechanism, for example:
- clustering or contiguity of fragile-and-drifting hits,
- normalized rank-based KL inside the low-margin subset,
- or mismatch-adjacent concentration measures that remain pre-final.
- If the current feature family has already saturated, the next requirement is likely not another threshold sweep but a larger corrected validation sample and/or new pre-final features.

### Next step
- Treat the current Phase 4 result as:
- a validated mechanistic candidate family,
- with a preferred precision-oriented operating point,
- but with insufficient evidence for the original `>=0.900` precision / `>=0.500` recall target.
- The next experiment should either:
- add richer accepted-prefix interaction summaries that preserve the current mechanism,
- or expand corrected validation data beyond `steps50` before claiming further improvement.

## SESSION HANDOFF — 2026-04-07 11:08:29 KST
### Current phase and sub-step
Phase 4 robustness-aware indicator design is complete for the current pre-final feature family.

### State of work
- New script exists:
- `tests/RetrievalAttention_new/research/phase4_indicator_design.py`
- New artifacts exist:
- `tests/RetrievalAttention_new/research/data/phase4_indicator_block_features.csv`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_token_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_rule_metrics.csv`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_shortlist.csv`
- New written summary exists:
- `tests/RetrievalAttention_new/research/phase4_summary.md`
- `tests/RetrievalAttention_new/research/scripts.md` has been updated.

### Critical context the next session MUST know
- Use `CUDA_VISIBLE_DEVICES=0`.
- Use `conda run -n retroinfer python ...` for profiling / scan runs.
- The accepted-prefix `early_margin + KL(early_verify || draft)` mechanism survived the Phase 4 robustness step.
- Accepted-prefix-only rules are deployable and reasonably stable, but in the useful recall regime they appear capped at about `0.72` precision on corrected `steps50`.
- Best accepted-prefix-only rule with `steps50` recall `>= 0.50`:
- `accepted_high_count|margin<0p05|count>=1`
- `steps50`: precision `0.723`, recall `0.550`
- Best robustness-aware deployable candidate:
- `source_or|accepted_high_count|margin<0p05|or|bonus_ge0_count>=2`
- `steps20`: precision `0.821`, recall `0.639`
- `steps50`: precision `0.824`, recall `0.514`
- Best high-recall candidate:
- `source_or|accepted_weighted_score|margin<0p05|or|bonus_ge0_count>=1`
- `steps20`: precision `0.659`, recall `0.806`
- `steps50`: precision `0.672`, recall `0.789`
- No tested Phase 4 rule reaches precision `>= 0.900` with recall `>= 0.500` on both datasets.

### Exact next action
- Either:
- design richer accepted-prefix block summaries that preserve the same mechanism,
- or collect a larger corrected validation sample beyond `steps50` before claiming further progress.

### Files to read first
- `tests/RetrievalAttention_new/research/research_log.md`
- `tests/RetrievalAttention_new/research/phase4_summary.md`
- `tests/RetrievalAttention_new/research/phase4_indicator_design.py`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_token_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_rule_metrics.csv`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_shortlist.csv`
