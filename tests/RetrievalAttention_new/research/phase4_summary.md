# Phase 4 Summary

## Goal

Phase 4 moved from mechanism explanation to deployable indicator design. The target was a robustness-aware pre-final rejection indicator family centered on the accepted-prefix interaction between:

- low `early_margin` as a fragility signal
- elevated `KL(early_verify || draft)` as a drift signal

The design constraint remained unchanged:

- only use features available before final verify
- do not use final-verify outputs in any deployable rule

## New Phase 4 artifacts

- `tests/RetrievalAttention_new/research/phase4_indicator_design.py`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_block_features.csv`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_token_summary.csv`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_rule_metrics.csv`
- `tests/RetrievalAttention_new/research/data/phase4_indicator_shortlist.csv`

## Mechanism carried forward

The accepted-prefix interaction remains the core valid online signal.

On accepted-prefix tokens, the dangerous regions are still the low-margin slices:

- `steps20`
  - `low_margin_only`: `6/39`, reject rate `15.38%`
  - `low_margin_plus_moderate_kl`: `6/46`, reject rate `13.04%`
  - `low_margin_plus_high_kl`: `5/21`, reject rate `23.81%`
  - `moderate_kl_without_low_margin`: `3/439`, reject rate `0.68%`
  - `high_kl_without_low_margin`: `1/168`, reject rate `0.60%`
- `steps50`
  - `low_margin_only`: `13/99`, reject rate `13.13%`
  - `low_margin_plus_moderate_kl`: `14/131`, reject rate `10.69%`
  - `low_margin_plus_high_kl`: `16/79`, reject rate `20.25%`
  - `moderate_kl_without_low_margin`: `9/1155`, reject rate `0.78%`
  - `high_kl_without_low_margin`: `10/526`, reject rate `1.90%`

So the core Phase 3 interpretation survives Phase 4:

- low margin is the necessary fragility context
- moderate KL matters, not only the extreme high-KL tail
- KL without low margin is mostly too safe to drive a clean block rule by itself

## Accepted-prefix-only rule families

Phase 4 searched accepted-prefix block summaries including:

- count of low-margin + high-KL rows
- count of low-margin + moderate/high-KL rows
- fraction of accepted-prefix rows in those categories
- weighted interaction scores where high-KL hits count more heavily than moderate-KL hits
- top-k / quantile summaries of KL among low-margin accepted-prefix rows

The main accepted-prefix-only result is that these features are stable and deployable, but capped in precision on `steps50`.

Best accepted-prefix-only rule with `steps50` recall at or above `0.50`:

- `accepted_high_count|margin<0p05|count>=1`
- `steps20`: precision `0.720`, recall `0.500`, F1 `0.590`
- `steps50`: precision `0.723`, recall `0.550`, F1 `0.625`

Best more explicitly interaction-style accepted-prefix-only rule:

- `accepted_weighted_score|margin<0p05|score=2*high+mod|>=3`
- score = `2 * count(low_margin + high_kl) + count(low_margin + moderate_kl)`
- `steps20`: precision `0.731`, recall `0.528`, F1 `0.613`
- `steps50`: precision `0.720`, recall `0.541`, F1 `0.618`

Best `steps20`-selected accepted-prefix-only rule did not validate cleanly:

- `accepted_top2_kl|margin<0p05|top2>=0.02`
- selected on `steps20`: precision `0.857`, recall `0.500`
- validated on `steps50`: precision `0.769`, recall `0.459`

This is the main Phase 4 accepted-prefix-only conclusion:

- the interaction family is real and reasonably stable
- but accepted-prefix interaction summaries alone still top out around `0.72` precision on corrected `steps50` once recall is kept around `0.5`

## Candidate deployable family

The strongest robustness-aware family kept the accepted-prefix interaction as the core branch and added a restrained early-bonus fragility branch only when it materially improved robustness.

Best precision on `steps50` among rules with recall at or above `0.50`:

- `source_or|accepted_high_count|margin<0p05|or|bonus_ge0_count>=2`
- trigger if either:
- at least `2` accepted-prefix rows satisfy `early_margin < 0.05` and `KL(early_verify || draft) >= 0.03`
- or at least `2` early-bonus rows satisfy `early_margin < 0.05` and `early_entropy > 3.5`

Metrics:

- `steps20`: precision `0.821`, recall `0.639`, F1 `0.719`, trigger rate `0.318`, `23 TP`, `5 FP`, `13 FN`
- `steps50`: precision `0.824`, recall `0.514`, F1 `0.633`, trigger rate `0.294`, `56 TP`, `12 FP`, `53 FN`

Why this is the current Phase 4 candidate:

- it is explicitly grounded in the accepted-prefix margin+KL interaction
- it preserves that interaction as the main precision branch
- it uses the early-bonus branch only because it materially improves robustness at the target recall regime
- it is remarkably stable in precision across datasets:
  - `0.821` on `steps20`
  - `0.824` on `steps50`

Relative to the best accepted-prefix-only rule that still keeps `steps50` recall above `0.50`:

- accepted-prefix-only: `0.723` precision / `0.550` recall on `steps50`
- source-aware candidate: `0.824` precision / `0.514` recall on `steps50`

So the bonus branch buys about `+0.10` absolute precision on corrected `steps50` while staying above the recall floor.

## Best high-recall family

If recall / F1 is prioritized over precision, the strongest family was:

- `source_or|accepted_weighted_score|margin<0p05|or|bonus_ge0_count>=1`
- `steps20`: precision `0.659`, recall `0.806`, F1 `0.725`
- `steps50`: precision `0.672`, recall `0.789`, F1 `0.726`

This family is useful as a high-recall operating point, but it is not the preferred deployable candidate because the precision cost is too large.

## Robustness conclusion

No tested Phase 4 rule reached the original target on both datasets:

- precision `>= 0.900`
- recall `>= 0.500`

The strongest current evidence is therefore:

- the accepted-prefix interaction family is sufficient to define a real deployable indicator direction
- it is not sufficient by itself to reach the target robustness level on corrected `steps50`
- adding a restrained early-bonus branch materially improves precision at the target recall regime
- even with that addition, the current pre-final feature family still appears capped below the original target

## Paper-ready design logic

The current Phase 4 design logic is:

1. Treat accepted-prefix low margin as the online fragility context.
2. Within that fragile subset, separate moderate/high `KL(early_verify || draft)` from low-KL rows because drift is only useful when the row is already brittle.
3. Aggregate those accepted-prefix interaction hits at the block level rather than firing on a single token.
4. Prefer repeated high-KL fragile hits for precision.
5. Add an early-bonus fragility branch only when it improves robustness on corrected `steps50`, not because it won on `steps20`.

## What is still needed next

The Phase 4 rule family is deployable, but the evidence says the current feature family is still insufficient for the original target. The next most precise follow-up would be one of:

- collect a larger corrected validation set beyond `steps50` to test whether the `~0.82` precision candidate is truly stable or still optimistic
- add richer pre-final accepted-prefix summaries that preserve the same mechanism but capture block concentration better than simple counts:
  - contiguous-hit structure
  - local clustering near mismatch-adjacent regions
  - normalized rank-based KL summaries within low-margin rows
- or explicitly model two deployable operating points:
  - a higher-precision escalation rule
  - and a broader high-recall caution rule

At the current evidence level, the cleanest statement is:

- accepted-prefix `early_margin + KL` is a real and robust mechanism
- it supports a usable deployable block indicator family
- but the present pre-final-only features do not yet support a `>=0.90` precision, `>=0.50` recall rule on corrected `steps50`
