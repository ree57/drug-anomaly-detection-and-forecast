# Step 8a: anomaly definition and evaluation plan

Built by `anomaly_eval.py` (+ `sql/eval_events.sql`) into `data/eval/eval_events/`: one row per row of
`price_events`. Counts and examples are printed by the script; they are not copied here so they can't go stale.

## What we call an anomaly

A **real** price change of a drug group that is **unusual for that group's own history and its class**
(generic vs brand), and is **not an expected, scheduled event**. Two types:

- **Level shift:** the price moves and stays near the new level.
- **Reverting spike:** the price moves and returns to the old level within 4 weeks.

Not anomalies: `mix_only` events (the median moved only because NDCs joined or left the group), January brand
list-price increases, and small moves inside the group's normal range.

We have **no ground-truth labels**, so the labels below are **proxies**. Positives use the future (`outcome_*`)
to say what a move turned out to be. That is fine for grading, but a detector must never see it.

## Categories (column `category`, first matching rule wins; `is_pos_*` / `is_neg_*` flags can overlap)

| Category | Rule | Uses the future? |
|---|---|---|
| `excluded_mix_only` | `mix_only` | no |
| `excluded_artefact` | `artefact_suspect` (group ≥ 30%, matched NDCs < 5%) or judged an artefact in manual review (`reports/top20_review.md`); reason in `artefact_reason` | no (auto) / review |
| `pos_big_spike` | \|pct_change\| > 50% and `outcome_reverted` | yes |
| `pos_big_shift` | \|pct_change\| > 50%, not reverted, 8 weeks later closer to the new price than the old | yes |
| `pos_relative` | not a brand January increase, at least 6 earlier real changes, \|pct_change\| ≥ 10% **and** ≥ 5 × the group's 95th percentile of its *earlier* \|changes\|, 8 weeks later closer to the new price | yes (persistence) |
| `big_unclear` | \|pct_change\| > 50% but outcome unknown yet or mixed: not scored | yes |
| `neg_january_list_increase` | brand, effective in January, +1% to +20%, within 0.1 points of a whole % | no |
| `neg_normal_move` | at least 6 earlier real changes, \|pct_change\| ≤ the group's 75th percentile of earlier \|changes\| and < 10% | no |
| `unlabelled` | everything else | none |

Notes:
- `pos_relative` excludes **every** brand January increase, not only the round ones, so a list-price increase
  that doesn't happen to be round (Otezla's +7.4% in 2022) can never count as a positive.
- `after_gap` events are kept and keep their flag.
- `prior_*` columns use a window that ends one row **before** the event: past information only.
- `split`: `train` = published before 2025, `test` = 2025–2026. Detectors are tuned on `train` and scored on `test`.

Named test cases (`test_case`, `test_case_expect`): isosorbide 20 mg, prucalopride 2 mg and phenobarbital 16.2 mg
moves over 50% should be **flagged**; Otezla 30 mg January increases should **not** be.

## Metrics (each reported separately for `pos_big_shift`, `pos_big_spike`, `pos_relative`)

Every detector gives each event a score; higher means more suspicious. On the `test` split:

1. **Recall@N:** the share of a positive type that lands in the detector's top N flags (N = 100, 500, 1,000).
   "If an analyst only checks N alerts, how many real anomalies do they see?"
2. **Expected-event false flags@N:** how many `neg_january_list_increase` and `neg_normal_move` events are in the
   top N. These are alerts we know are noise.
3. **Known-positive share@N:** the share of the top N that are known positives. This is a lower bound on
   precision, because `unlabelled` events may be real anomalies too.
4. **Average precision per positive type:** one number summarising ranking quality over the labelled events only
   (positives of that type vs negatives), so it doesn't depend on picking N.
5. **Named test cases:** pass/fail. The `flag` events must rank in the top 1% of scores, and Otezla's January
   increases must stay out of the top 10%.
6. **Manual review of the top 20:** read each flag and call it real level shift, real spike, expected event,
   or data issue. This catches what the proxy labels can't.

**Baseline every detector must beat: rank by \|pct_change\|.** By construction it catches the >50% positives
almost perfectly, so the real test is `pos_relative` (moves big for *that* drug but not big in absolute terms)
and keeping expected events out of the top N.

## Decisions after 8a (applied from step 8b, in `evaluate.py`)

- **Reverted spikes are scored on ALL years and are descriptive only:** the test split has a single one. We report
  each spike's percentile rank among all real events (median, how many in the top 1% and top 5%).
- **Detectors are judged mainly on expected-event false flags@N and the manual top-20 review.** Recall and
  average precision are supporting numbers.
- **Clusters:** events with the same published week and the same `pct_change` rounded to 0.1 count as one
  cluster. The cluster's score is its highest member's score. A cluster is positive if any member is, negative if
  none is positive and at least one is negative. Every metric is reported for events and for clusters. Rounding also
  merges unrelated small moves, so negative cluster counts are approximate.
- Ranking pool: real events only (`mix_only` is known at publication, so a detector may skip those events).

## Step 8b-2 additions

- **Pool:** events whose `category` starts with `excluded_` (mix_only, artefacts) are not ranked. Detectors also
  drop `artefact_suspect` events from their history.
- **Hard negatives** (past information only; big in absolute terms but expected):
  - `neg_hard_january`: a January brand list increase (round %) of +7% to +20%. A subset of
    `neg_january_list_increase`.
  - `neg_hard_volatile`: \|change\| between 20% and 50%, no bigger than the group's own 75th percentile of earlier
    changes (at least 6 earlier changes), and not a positive.
  - New metrics: hard false flags@N for each type, and average precision with only hard negatives as the
    negatives.
- **Families** replace the "same week + rounded %" clusters (`build_families` in `evaluate.py`). Two events of the
  same week are linked if they share the first word of the description (split on space and '-'), or, for moves of
  30% or more, the same % change to 2 decimals. Chains of links form one family (iterative smallest-label
  propagation). A family is positive if any member is, and a known negative only if every member is a known
  negative.
  - Tried first: 10% / 1 decimal. Coincidental equal % changes among generics chained 2,112 events into one family.
  - Known gaps: store-brand prefixes (`FT`, `GNP`, `GS`...) are treated as a name, so a store's many OTC products
    of one week become one family (largest: 312). Same-company actions under different names (Fiasp vs Tresiba,
    different %) stay separate.
- **Hybrid detector** (`detect_hybrid.py`): the average of two percentiles against the train period (2022–24),
  |ln(new/old)| and variant B's |z|. A tie-breaker below 1e-6 orders events above the reference maximum.

Caveat: `pos_relative` is itself defined by "size compared with the group's own past", so a detector that scores
exactly that ratio will look very good on it. Detectors should be judged on all metrics together, and the manual
top-20 review is the check against "grading our own homework".
