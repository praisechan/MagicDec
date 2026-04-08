# Phase-Based Research: Final-Verify Rejection Indicator for 3-Stage Speculative Decoding

You are working in /home/juchanlee/MagicDec.

## Objective

Design an indicator that predicts, BEFORE final verify, whether a token block
will be rejected at final verify. Target: >=90% precision, >=50% recall.

Hard constraint: the indicator must use only information available before
final verify runs. Any feature that depends on final-verify outputs, final
settled tokens, final logits, KL to final verify, or top-k overlap with final
verify is not a valid online indicator. Those signals may be used only for
offline diagnosis and hypothesis formation.

## Why this matters

In the 3-stage hierarchical self-spec pipeline (draft at budget1 → early verify
at budget2 → final verify at full attention), final verify uses full attention
and is expensive. Tokens that pass early verify (retrieval-attention with budget2)
but fail final verify (full attention) represent a gap between approximate and
exact attention. If we can predict which blocks will fail final verify, we can
selectively escalate only those blocks to a higher budget before wasting a
full-attention call.

## What has already been tried (and limitations)

Previous work profiled these signals but did NOT achieve 90%+ precision with 50%+ recall:
- Draft confidence: top1/top2 prob, margin
- Early-verify confidence: top1/top2 prob, margin

Key existing infrastructure:
- `tests/RetrievalAttention_new/profile_margin_final_verify.py`: margin vs rejection histograms
- `tests/RetrievalAttention_new/selfspec_benchmark.py`: full 3-stage loop with CSV logging

These approaches failed because they treated the problem as threshold-tuning on
shallow statistics. They did not investigate the MECHANISM of why retrieval
attention at budget2 produces different outputs from full attention on specific
tokens.

## Required reading before any work

Read these files in this order to build a precise mental model:
1. `/home/juchanlee/MagicDec/prompts/retrievalattention_new_3stage_implementation_summary.md`
2. `/home/juchanlee/MagicDec/Engine/RetrievalAttention_new/backend.py`
3. `/home/juchanlee/MagicDec/tests/RetrievalAttention_new/selfspec_benchmark.py`
4. `/home/juchanlee/MagicDec/Engine/RetrievalAttention_new/cache_hub/retroinfer_cache.py`
5. `/home/juchanlee/MagicDec/Engine/RetrievalAttention_new/model_hub/llama.py`

---

## Phase 1: Deep Understanding (do this FIRST, thoroughly)

Answer these questions for yourself before proceeding to any other phase:

- What exactly does RetroInfer retrieval do differently from full attention?
  Trace the code path from `_activate_cache` through the actual attention
  computation.
- What information is LOST when using budget2 instead of full attention?
  Which KV entries are dropped, and how is the selection made?
- When early verify accepts a token, what does that actually mean about the
  probability distribution? (It means the argmax matches, but what about the
  shape of the distribution?)
- When final verify rejects, what changed? Finding the difference between rejected one and accepted one is the goal.

Checkpoint: Write a summary (under 300 words) of your understanding to
`research/phase1_summary.md` and log your findings to `research/research_log.md`
before moving to Phase 2.

---

## Phase 2: Mechanistic Investigation (Hypothesis + Empirical Probing)

This phase is NOT pure thinking. You must write and run probing scripts to look
inside the model's intermediate representations. Form hypotheses, then immediately
test them with data before moving on.

### 2a. Hypothesis generation

Reason about WHAT MAKES rejected tokens different. Don't just list metrics —
reason about the information-theoretic and computational causes.

Key questions to investigate:

1. **Retrieval approximation error**: When RetroInfer with budget2 retrieves
   KV entries, which entries does it miss? Are rejected tokens the ones where
   the missed entries were important for the correct prediction?

2. **Distribution shape vs argmax**: Early verify checks argmax agreement.
   But two distributions can agree on argmax while having very different shapes.
   Does final verify rejection correlate with cases where early-verify
   distribution is "fragile" — high argmax agreement but low distributional
   similarity?

3. **Positional effects**: Does rejection concentrate at specific positions
   within the gamma2 block? Beginning (where early-verify cache state might be
   slightly stale), end (where accumulated drift matters most), or at the bonus
   token position?

4. **Attention pattern sensitivity**: Some tokens depend on a few critical KV
   entries (e.g., named entities, numbers, rare words). Others depend broadly
   on many entries. Retrieval budget affects the former more. Can we detect this
   from the logit distribution (e.g., entropy pattern, top-k concentration)?

For each hypothesis, identify:
- What observable signal would confirm it
- What data you need to collect
- What existing logged features relate to it (vs what's missing)

### 2b. Empirical probing — logit-level inspection

Write a probing script that runs a small number of blocks (10-20 settlement
cycles) through all 3 stages and captures, FOR EACH TOKEN POSITION:

- Full logit vectors from draft, early verify, and final verify
- Softmax distributions from all 3 stages
- Token IDs: argmax from each stage, ground truth from final verify
- Per-position: KL(final_verify || early_verify), KL(final_verify || draft)
- Per-position: JS divergence, total variation distance
- Top-10 token overlap between early_verify and final_verify
- Rank of final_verify's top-1 token in early_verify's distribution
  (if early verify ranks the correct token at #3 instead of #1, that's
  a very different failure than ranking it at #847)

Compare these between ACCEPTED positions and REJECTED positions (the mismatch
position and its neighbors). Look for systematic patterns, not just averages.

Save output to `research/data/phase2b_logit_probe_<model>_<prefix_len>.csv`.

### 2c. Empirical probing — hidden state inspection

If logit-level analysis is insufficient, probe deeper:

- Extract hidden states from the LAST transformer layer before the LM head,
  for both early verify and final verify, at each token position.
- Compute cosine similarity between early-verify and final-verify hidden states
  at each position. Does the rejected position show lower similarity?
- Compute the L2 norm of the hidden state difference. Is it larger at rejected
  positions?
- Check attention weights (if accessible from the cache or model internals):
  at the rejected position, does early verify attend to different KV entries
  than final verify would? This directly measures the retrieval approximation
  error.

Save output to `research/data/phase2c_hidden_state_probe_<model>_<prefix_len>.csv`.

### 2d. Empirical probing — budget sensitivity test

Run the same block through early verify at 3 different budgets:
budget2 * 0.8, budget2, budget2 * 1.2.

For each position, check:
- Does the argmax token change across budgets? (= retrieval-sensitive position)
- Does the margin change significantly? (= confidence is budget-dependent)
- Do positions that are retrieval-sensitive correlate with final-verify rejection?

This directly tests whether "retrieval approximation error" is the dominant
failure mode, and whether sensitivity to budget is a usable signal.

Save output to `research/data/phase2d_budget_sensitivity_<model>_<prefix_len>.csv`.

### 2e. Synthesis

After running 2b-2d, update your hypotheses:
- Which hypotheses were confirmed, refuted, or inconclusive?
- What new patterns emerged that you didn't predict?
- Which observable signals showed the clearest separation between accepted
  and rejected tokens?

Write findings to `research/research_log.md` AND to `research/phase2_summary.md`
before proceeding to Phase 3.

Checkpoint: You must have run at least 3 probing scripts and collected real data
before exiting Phase 2. Pure reasoning without code execution is NOT sufficient
to complete this phase.

---

## Phase 3: Large-Scale Empirical Investigation

Phase 2 probed a small number of blocks for deep insight. Phase 3 collects
data at scale to validate patterns and compute statistics.

### 3a. Rejection anatomy (at scale)

For each final-verify rejection across a full profiling run (not just 10-20 blocks):
- Was the mismatch on a drafted token or the early-verify bonus token?
- What was the mismatch position within the block?
- What was early-verify's confidence (margin, entropy) at the mismatch position?
- What was draft's confidence at the mismatch position?
- How did the full early-verify distribution compare to full-attention
  distribution at the mismatch position?

### 3b. Distribution comparison at scale

Based on what you learned in Phase 2, collect the most promising signals at
scale. At minimum, for every token position in every block:

- Draft and early-verify top-10 logits (probs + token IDs)
- Draft and early-verify margin, entropy
- KL(early_verify || draft) per position
- Top-k overlap (Jaccard of top-5 and top-10) between draft and early verify
- Rank of draft's top-1 in early-verify's distribution
- Any new signals identified in Phase 2 probing
- Accepted/rejected flag for the block
- Position of first mismatch (if rejected)
- Whether mismatch is on a bonus token

### 3c. Profiling script

Write a profiling script (new file, do not modify existing) that instruments
one full 3-stage run and logs all the above. Output as a single detailed CSV.

Run with at least 2 different configurations (different models or prefix lengths).

Save outputs to `research/data/phase3_full_profile_<model>_<prefix_len>.csv`.

### 3d. Pattern analysis

Analyze the collected data:
- Are there distinct clusters of rejection cases, or one continuous pattern?
- Do bonus-token rejections have a different signature from drafted-token rejections?
- Does rejection rate vary by position within the block?
- Are there interactions between features (e.g., low margin + high entropy
  delta predicts rejection but neither alone does)?
- What is the distribution of each candidate signal for accepted vs rejected blocks?

### 3e. Valid-indicator filter

Before naming any "best indicator," explicitly filter candidate signals into:

- valid pre-final features:
  draft-only features, early-only features, and draft-vs-early features
- invalid online features:
  anything that uses final verify outputs directly

If an offline diagnostic looks strong but depends on final verify, say so
clearly and do not present it as a deployable indicator.

---

## Current status and exact next step

Phase 1, Phase 2, and the first Phase 3 aligned run are complete.

Existing aligned Phase 3 artifact:
- `/home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768.csv`

Existing summaries:
- `/home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase2_summary.md`
- `/home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_summary.md`
- `/home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/research_log.md`

Important interpretation update:
- `KL(final||early)` looked strong in the first Phase 3 run, but it is NOT a valid online indicator because it requires final verify.
- The current best valid pre-final candidate is an early-only fragility rule based on low early margin and high early entropy.

Your next task is to continue Phase 3 under the pre-final-only constraint.

Specifically:
1. Read the latest `SESSION HANDOFF` in `/home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/research_log.md`.
2. Read `/home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_summary.md`.
3. Use only the `retroinfer` environment for runs:
   `conda run -n retroinfer python ...`
4. Do not modify production code paths; only add research artifacts if needed.
5. Stay in Phase 3 and collect more aligned data with the existing profiler:
   `/home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_full_profile.py`
6. Evaluate only pre-final candidate indicators, using features available before final verify, such as:
   - early margin
   - early entropy
   - draft margin / entropy
   - draft-vs-early top-k overlap
   - draft-vs-early rank shifts
   - source kind (`accepted_prefix` vs `early_bonus`)
   - position within block
   - verify mode (`normal` vs `high`)
7. Do NOT treat any final-based feature as a deployable indicator.
8. Append findings to:
   `/home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/research_log.md`
9. Update:
   `/home/juchanlee/MagicDec/tests/RetrievalAttention_new/research/phase3_summary.md`

Concrete next analysis goal:
- test whether a pre-final fragility gate built from early-only and draft-vs-early features can retain high precision while improving recall beyond the current tiny-sample rule:
  `early_margin < 0.1 and early_entropy > 3.5`

Deliverable expectation for the next step:
- identify the best CURRENT valid pre-final indicator candidate,
- report its precision/recall on the expanded Phase 3 collection,
- and clearly separate online-usable indicators from offline-only diagnostics.

Checkpoint: Report raw findings with numbers. What patterns do you see?
Write to `research/research_log.md` and `research/phase3_summary.md`.

---

## Phase 4: Indicator Design

Based on Phase 2 and 3 findings, design candidate indicators. Requirements:
- Must be computable from information available BEFORE final verify
  (draft logits + early-verify logits + accepted prefix length + position info)
- Must be simple enough to hard-code (no learned models)
- Target: >=90% precision, >=50% recall

Evaluate ALL of these candidate categories (don't just pick one prematurely):

1. **Single scalar threshold**: best single feature with threshold
2. **Feature ratio/combination**: e.g., margin_early / margin_draft, or
   entropy_delta * (1 / margin_early)
3. **Position-aware rule**: different thresholds for bonus token vs drafted tokens
4. **Top-k overlap score**: Jaccard similarity of top-k between draft and early verify
5. **Worst-position rule**: flag block if ANY position exceeds threshold
   (vs average across block)
6. **Two-stage filter**: first filter on cheap signal, then check expensive signal
   only for borderline cases
7. **Any new indicator types suggested by Phase 2/3 findings**

For each candidate:
- Compute precision, recall, F1, trigger rate, FPR, FNR
- Compute at MULTIPLE operating points (don't just report the best one)
- Compare against baselines: "always escalate" and "never escalate"
- Report separately for bonus-token rejections vs drafted-token rejections

Save evaluation results to `research/data/phase4_indicator_evaluation.csv`.

---

## Phase 5: Recommendation

Produce a final recommendation with:

1. The single best indicator (or smallest rule set) meeting the precision/recall target
2. Clear explanation of WHY it works — linked back to the mechanistic
   understanding from Phase 2
3. Exact implementation specification: thresholds, feature computation,
   decision rule — ready to hard-code
4. Failure cases: when does this indicator miss rejections or false-alarm?
5. Whether additional feature logging would improve results
6. A comparison table of ALL evaluated candidates with:
   - precision, recall, F1, trigger rate, FPR, FNR
   at multiple operating points

Write the final report to `research/final_recommendation.md`.

---

## Constraints

- Do not break existing code paths. Create new experimental scripts only.
- Do not overwrite existing logs or data files.
- Do not use previously deleted files. They were removed because they did not work.
- All new scripts go in `tests/RetrievalAttention_new/research/` or subdirectories.
- Prefer interpretable heuristics. No ML models.
- Be VERY skeptical of metrics that look good only at tiny trigger rates.
- Be skeptical of metrics that only beat baseline by sacrificing too much recall
  unless the precision gain is clearly worth it.
- Explain WHY final-verify rejection happens, not just which score correlates with it.
- Do not skip any phase. The depth of Phase 2 and 3 is what was missing in
  previous attempts.
- When running experiments, use theses commands settings.
  CUDA_VISIBLE_DEVICES=0 \
  python tests/RetrievalAttention_new/<python filename>.py \
    --dataset pg19 \
    --model_name Meta-Llama-3.1-8B \
    --prefix_len 32768 \
    --num_max_token 100 \
    --gamma1 6 \
    --gamma2 32 \
    --budget1 0.02 \
    --budget2 0.10 \
    --budget2_high <can be changed> \
    --enable_dynamic_budget \
    --T_low <can be changed> \
    --T_high 0.2
---

## Cross-Session Logging Protocol

Each phase may span multiple sessions. To ensure continuity, follow these rules
strictly:

### Research log

Maintain a running log at `tests/RetrievalAttention_new/research/research_log.md`.

After EVERY meaningful step (not just phase boundaries), append an entry:

```
## [Phase X.Y] <short title> — <date/time>
### What was done
<concrete actions: scripts run, files read, data collected>
### Key findings
<bullet points of observations, with numbers>
### Open questions
<what remains unclear>
### Next step
<exact next action to take>
```

### Data artifacts

All profiling outputs go to `tests/RetrievalAttention_new/research/data/`.

Use descriptive filenames:
- `phase2b_logit_probe_llama8b_8k.csv` — not `output.csv`
- `phase3_full_profile_llama8b_16k.csv` — not `data.csv`

Never overwrite. If re-running, append a suffix (`_v2`, `_v3`).

### Script registry

When you create a new script, add a one-line entry to
`tests/RetrievalAttention_new/research/scripts.md`:

```
- <filename.py> — <what it does> — <how to run it> — <which phase it belongs to>
```

### Phase completion marker

When a phase is complete, write a summary file:
`tests/RetrievalAttention_new/research/phase{N}_summary.md`

containing:
- Conclusions reached
- Data file locations and what each file contains
- The decision or understanding that gates entry to the next phase

### Session handoff (CRITICAL)

At the END of every session — when you are about to stop, run out of context,
or the user ends the conversation — write a handoff note to the research log:

```
## SESSION HANDOFF — <date/time>
### Current phase and sub-step
<e.g., "Phase 2c — hidden state probing, script written but not yet run">
### State of work
<what's done, what's in progress, what's blocked>
### Critical context the next session MUST know
<decisions made, hypotheses confirmed/rejected, surprises found>
### Exact next action
<the literal first thing to do next session — be specific>
### Files to read first
<paths the next session should read before doing anything>
```

### Starting a new session

When resuming work in a new session:
1. Read `tests/RetrievalAttention_new/research/research_log.md` (bottom-first
   for latest handoff)
2. Read the latest `tests/RetrievalAttention_new/research/phase{N}_summary.md`
3. Read any data files referenced in the handoff
4. Then continue from the exact next action specified

Do NOT re-read all source files or redo completed phases unless the handoff
note says something was inconclusive or needs revisiting.
