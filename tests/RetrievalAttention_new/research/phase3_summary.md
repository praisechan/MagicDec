# Phase 3 Summary

## What Phase 3 added

Phase 3 introduced an aligned full-profile profiler that keeps a per-position ledger of draft and early-verify features for every token that actually enters the unsettled online span before final verify. This fixes the main Phase 2b failure mode: the true final mismatch position no longer falls off the end of the last same-cycle early/draft window.

The new artifact is:

- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768.csv`
- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps5_cycles25.csv`

For the first real 32k run, the profiler produced 135 rows across 5 settled cycles. All 130 non-bonus settled positions had attached pre-final features. Only the 5 final authoritative bonus rows lacked pre-final features, which is expected because those tokens are created only by final verify.

## First findings

The alignment fix materially changed what we can say about rejection anatomy. The two true mismatch rows are now directly visible and both carry pre-final signals:

- cycle 2, position 8: source kind `early_bonus`, verify mode `high`
- cycle 4, position 16: source kind `accepted_prefix`, verify mode `normal`

These mismatch rows were not extreme “different-distribution” outliers in the Phase 2b sense. Their early-vs-final top-10 overlap was still `9/10` in both cases. The key separation was instead local fragility:

- exact mismatch rows: mean early margin `0.0496`, mean early entropy `3.8053`, mean KL(final||early) `0.0315`
- accepted non-bonus rows: mean early margin `0.4173`, mean early entropy `2.0398`, mean KL(final||early) `0.0064`

So the first aligned Phase 3 evidence points to a “same shortlist, wrong local ordering” failure mode. Final verify is not discovering a completely different candidate set at the mismatch position; it is re-ranking a highly uncertain local competition.

The rows immediately before rejection did not show the same degree of collapse in this logit-level view. At relative position `-1`, the two rejected blocks still had mean early margin `0.7650` and mean early entropy `1.0984`. That makes the logit-space signature look sharply localized at the mismatch position itself, even though Phase 2 hidden-state drift suggested representational distortion can begin earlier.

## Candidate indicator direction

This first run is small, but it gives a more concrete indicator direction than Phase 2:

- `early_margin < 0.1` and `early_entropy > 3.5` captured both observed mismatch rows.
- `early_margin < 0.1` and `KL(final||early) > 0.01` also captured both mismatch rows, while firing on `4/72` accepted non-bonus rows in this run.

However, the actual deployment goal is to predict rejection before final verify runs. That means any rule that uses final-verify outputs directly, including `KL(final||early)`, top-k overlap with final verify, or any feature derived from the final settled token, is not a valid online indicator. Those signals remain useful for offline diagnosis only.

So the best valid pre-final candidate from Phase 3 so far is:

- `early_margin < 0.1` and `early_entropy > 3.5`

This rule is weaker than the KL-based offline diagnostic, but it uses only early-verify information that exists before final verify. The next step is to search for stronger pre-final-only combinations using early logits, entropy shape, draft-vs-early disagreement, top-k structure, source kind, and block-position context.

These conclusions are still not ready to be treated as stable thresholds because the sample only contains 2 exact mismatch rows. But Phase 3 now has the right logging substrate to evaluate them properly on larger collections.

## Expanded GPU0 run

Using `CUDA_VISIBLE_DEVICES=0`, Phase 3 was extended to a larger aligned collection:

- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps5_cycles25.csv`

This run produced 664 rows across 20 settled blocks from 5 eval steps, with 8 rejected blocks and 8 directly observed mismatch rows. The mismatch rows continued to show extremely low early margin and high early entropy, but the expanded sample changed the ranking of the best valid online indicator.

The original early-only fragility gate degraded on the larger sample:

- `early_margin < 0.1 and early_entropy > 3.5`
- precision `0.500`, recall `1.000` at the block level (`8 TP`, `8 FP`, `0 FN`)

The best current valid pre-final candidate on the expanded GPU0 run is now:

- `early_margin < 0.03 and KL(early_verify || draft) > 0.01`

This is still fully pre-final because it uses only draft and early-verify distributions. On the expanded block-level scan it achieved:

- precision `0.700`
- recall `0.875`
- F1 `0.778`
- trigger rate `0.500`
- `7 TP`, `3 FP`, `1 FN`

The best overlap-based rule was weaker:

- `early_margin < 0.10 and top10_overlap_draft_early <= 8`
- precision `0.625`, recall `0.625`

So the strongest online signal so far is not pure uncertainty alone. It is a hybrid of:

- very low early-verify local certainty, and
- strong draft-vs-early distribution drift before final verify

This is a useful mechanistic update: the most predictive online cases are the ones where early verify is both fragile and already substantially disagreeing with the draft distribution, even before full attention is consulted.

The current candidate still misses the target of `>=90%` precision with `>=50%` recall, so Phase 3 remains incomplete. But it is a clear improvement over the earlier entropy-only rule and is the best current valid pre-final indicator candidate.

## Corrected steps20 run

Phase 3 was then extended to the requested larger setting:

- `num_eval_steps=20`
- `num_max_token=100`
- `CUDA_VISIBLE_DEVICES=0`

During that run, a profiling bug became visible: `max_cycles` had been applied globally across the entire run instead of per eval step. The research profiler was corrected so that `max_cycles` resets for each step, and the corrected artifact was saved separately:

- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`

The earlier pre-fix `steps20` artifact should be treated as a partial run:

- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps20_cycles25.csv`

The corrected per-step run produced:

- 2715 rows
- 88 settled blocks
- 36 rejected blocks
- 36 directly observed mismatch rows
- step coverage across all requested eval steps `0-19`

There was also one tokenizer warning for an overlength sample (`133585 > 131072`), but the run still completed and logged results through all 20 requested eval steps.

On this larger corrected sample, the rule ranking shifted again. The best precision among the scanned pre-final rules at or above `50%` recall became:

- `early_margin < 0.03 and KL(early_verify || draft) > 0.02`
- precision `0.783`
- recall `0.500`
- F1 `0.610`
- `18 TP`, `5 FP`, `18 FN`

The best F1 / more balanced valid pre-final rule in the current scan was:

- `early_margin < 0.10 and KL(early_verify || draft) > 0.02`
- precision `0.667`
- recall `0.833`
- F1 `0.741`
- `30 TP`, `15 FP`, `6 FN`

The best overlap-based rule on the corrected steps20 run was:

- `early_margin < 0.10 and top10_overlap_draft_early <= 8`
- precision `0.700`
- recall `0.875`
- F1 `0.778`

This larger run strengthens the same qualitative conclusion:

- pure early-only uncertainty is too broad,
- draft-vs-early disagreement is genuinely useful,
- and the best current pre-final indicators are hybrid rules combining low early margin with draft-vs-early divergence or overlap collapse.

It also changes the source mix of observed mismatch rows. In the corrected steps20 sample:

- 21 mismatch rows were `accepted_prefix`
- 15 mismatch rows were `early_bonus`
- 31/36 mismatch rows came from `high` verify mode

So rejection is not dominated by bonus tokens alone. Accepted-prefix failures are at least as important in the larger sample, and `high` mode is where most observed mismatches concentrate.

At that point in Phase 3, no scanned rule yet met the target of `>=90%` precision with `>=50%` recall. The next needed rule families were:

- source-aware rules,
- position-aware rules,
- and block-level aggregations stronger than a simple any-position trigger.

## Source-aware multi-hit scan

Phase 3 was then extended with a dedicated advanced pre-final scan on the corrected steps20 artifact:

- `tests/RetrievalAttention_new/research/phase3_advanced_pre_final_scan.py`
- `tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep.csv`
- `tests/RetrievalAttention_new/research/data/phase3_advanced_indicator_scan_Meta-Llama-3.1-8B_32768_steps20_cycles25_perstep_details.csv`

This scan searched the next rule family that the earlier summary said was still missing:

- source-aware rules that split `accepted_prefix` from `early_bonus`
- position-aware bonus rules
- count-based multi-hit block rules
- piecewise hybrids using low early margin plus draft-vs-early KL / overlap

That rule family produced a materially stronger valid pre-final indicator than the earlier best single-hit KL rule.

The new best valid pre-final block rule on the corrected steps20 artifact is:

- trigger if either:
- at least `2` `accepted_prefix` rows satisfy `early_margin < 0.05` and (`KL(early_verify || draft) > 0.03` or `top10_overlap_draft_early <= 7`)
- or at least `2` `early_bonus` rows at position `>= 4` satisfy `early_margin < 0.03` and `early_entropy > 3.8`

Metrics on the corrected steps20 artifact:

- precision `0.909`
- recall `0.556`
- F1 `0.690`
- `20 TP`, `2 FP`, `16 FN`

This is still a valid online indicator because it uses only features available before final verify:

- source kind
- position
- early margin / entropy
- and draft-vs-early KL / overlap on accepted-prefix rows

It does not use final-verify logits, final tokens, or any final-derived divergence metric.

This is the first Phase 3 rule on the corrected steps20 artifact that clears the target of:

- precision `>= 0.900`
- recall `>= 0.500`

The mechanistic lesson is that rejection is better modeled as repeated weakness inside a source-specific subfamily than as a single fragile row anywhere in the block.

- Accepted-prefix failures are best captured by repeated low-margin draft-vs-early drift or shortlist collapse.
- Early-bonus failures are best captured by repeated late-position low-margin high-entropy instability.

Each source-specific subrule was high precision but too low recall on its own:

- accepted-prefix branch alone: precision `0.929`, recall `0.361`
- early-bonus branch alone: precision `0.917`, recall `0.306`

Their OR-combination is what crosses the target by covering complementary rejection subfamilies.

Phase 3 is therefore no longer blocked on the original “any-position” rule family. The next step is no longer more threshold tweaking on the same sample; it is robustness testing of this source-aware multi-hit rule on additional corrected data or a holdout shard.

## Steps50 robustness check

Phase 3 was then extended to a larger corrected validation run:

- `tests/RetrievalAttention_new/research/data/phase3_full_profile_Meta-Llama-3.1-8B_32768_steps50_cycles25_perstep.csv`

This run produced:

- 7163 rows
- 231 settled blocks
- 109 rejected blocks
- 109 directly observed mismatch rows

The main result of this larger validation sample is that the earlier `steps20` win did not fully hold up.

The old simple `steps20` best rule:

- `early_margin < 0.03 and KL(early_verify || draft) > 0.02`

degraded on `steps50` to:

- precision `0.714`
- recall `0.550`
- `60 TP`, `24 FP`, `49 FN`

The old source-aware `steps20` winning rule:

- at least `2` `accepted_prefix` rows satisfy `early_margin < 0.05` and (`KL(early_verify || draft) > 0.03` or `top10_overlap_draft_early <= 7`)
- or at least `2` `early_bonus` rows at position `>= 4` satisfy `early_margin < 0.03` and `early_entropy > 3.8`

degraded more sharply on `steps50` to:

- precision `0.824`
- recall `0.385`
- `42 TP`, `9 FP`, `67 FN`

So the earlier `>=0.900` precision / `>=0.500` recall result should now be treated as an on-sample `steps20` win, not as a stable Phase 3 conclusion.

On the larger `steps50` sample, the best simple 2-metric rule from the scanned family became:

- `early_margin < 0.03 and KL(early_verify || draft) > 0.01`
- precision `0.688`
- recall `0.688`
- F1 `0.688`

The best advanced source-aware scanned rule on `steps50` became:

- trigger if either:
- at least `2` `accepted_prefix` rows satisfy `early_margin < 0.05` and (`KL(early_verify || draft) > 0.03` or `top10_overlap_draft_early <= 7`)
- or at least `2` `early_bonus` rows satisfy `early_margin < 0.05` and `early_entropy > 3.5`

with:

- precision `0.824`
- recall `0.514`
- F1 `0.633`
- `56 TP`, `12 FP`, `53 FN`

This larger run keeps the same qualitative lesson:

- low `early_margin` plus draft-vs-early drift is still useful
- source-aware multi-hit logic is still better than a single-token trigger

But it changes the quantitative conclusion:

- no scanned rule on `steps50` still reaches the target of `>=0.900` precision with `>=0.500` recall

So Phase 3 should now be viewed as having identified a promising rule skeleton, but not yet a threshold set that has proven robust at larger scale. The next step is robustness-aware rule design rather than more confidence in the `steps20` thresholds.

## Why margin and KL help together

To answer the mechanistic question directly, Phase 3 was extended with a dedicated accepted-prefix interaction analysis:

- `tests/RetrievalAttention_new/research/phase3_interaction_analysis.py`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_conditional_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_block_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_mechanism_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_heatmaps.png`
- `tests/RetrievalAttention_new/research/data/phase3_interaction_accepted_prefix_conditional_lines.png`

This analysis stayed strictly pre-final on the feature side:

- `early_margin`
- `KL(early_verify || draft)`
- source kind
- per-block counts of those pre-final conditions

The final settled mismatch labels were used only for offline evaluation, not as an online feature.

The clearest source-aware conclusion is that KL is mostly useful as a *contextual* signal, not as a standalone one. On accepted-prefix rows in `steps50`:

- overall accepted-prefix token reject rate was only `1.11%`
- high KL without low margin had reject rate `1.90%` (`10/526`)
- moderate KL without low margin had reject rate only `0.78%` (`9/1155`)
- low margin alone had reject rate `13.13%` (`13/99`)
- low margin plus moderate KL had reject rate `10.69%` (`14/131`)
- low margin plus high KL had reject rate `20.25%` (`16/79`)

So the old KL-only histogram looked weak for a real reason: most KL mass sits in regions where early verify is not actually fragile enough for that drift to be dangerous. In the `steps50` high-KL slice:

- low-margin rows had reject rate `20.25%` (`16/79`)
- mid-margin rows had reject rate `4.43%` (`7/158`)
- high-margin rows had reject rate `0.82%` (`3/368`)

That pattern is also visible in the new heatmap figure: the risky region is the lower-left margin corner combined with moderate-to-high KL, while large high-margin / high-KL areas stay mostly harmless.

The block-level view explains why the combination still helps even though KL alone is weak. On accepted-prefix blocks in `steps50`:

- any low-margin row: precision `0.654`, recall `0.817`
- any moderate-KL row: precision `0.590`, recall `0.936`
- any low-margin + moderate-KL row: precision `0.688`, recall `0.505`
- any low-margin + high-KL row: precision `0.721`, recall `0.404`
- at least 2 low-margin + high-KL rows: precision `0.938`, recall `0.138`

This is the main mechanism:

- low margin identifies locally fragile rows, but many of them are harmless
- KL alone identifies draft-vs-early drift, but much of that drift occurs on confident rows and is therefore cheap / harmless
- when both appear together, the row is both fragile *and* already disagreeing with the draft distribution, so reject likelihood rises sharply
- repeated joint hits inside a block isolate a smaller but much purer rejection family

The moderate-KL regime matters because it is where much of the usable recall lives. In `steps50`, among blocks that already contain a low-margin accepted-prefix row:

- the subset reject rate is `65.4%` (`89/136`)
- requiring at least one low-margin + moderate-KL hit keeps `68.8%` precision and still covers `55/89` of those rejected low-margin blocks
- requiring at least one low-margin + high-KL hit is cleaner at `72.1%` precision, but it covers only `44/89`

So the interaction is not only an extreme-tail story. Moderate KL becomes informative *once margin says the row is already brittle*.

The stable paper-ready explanation is:

- `early_margin` is a fragility signal.
- `KL(early_verify || draft)` is a drift signal.
- Drift without fragility is often harmless.
- Fragility without drift is broader and noisier.
- The useful accepted-prefix rejection family is the intersection: locally brittle rows where early verify has already moved away from the draft distribution before final verify runs.

This explains why `early_margin + KL` beats either metric alone, while also explaining why the exact `steps20` thresholds did not survive unchanged on `steps50`: the interaction is real, but its calibration is heterogeneous enough that a single threshold pair is not yet robust.
