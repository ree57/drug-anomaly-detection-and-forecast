# Step 10c: competition features in the move classifier (one run)

Generated with `competition_features.py` (features and leakage checks) and `forecast_competition.py` (signal check,
with/without comparison, watch list). All numbers are also in `reports/metrics_competition.csv` and
`reports/metrics_competition_validation.csv`.

We have seen test results in Steps 10 and 10b, so this is again an honest follow-up:
- the choice between versions uses **validation only**
- the features and design were fixed before the run, and it was run once
- the only re-run fixed a printing error, after the validation choice had already been printed

## Features (past-only, from the NDC labeler code = first 5 digits)

**Definition.** The set of labelers among a group's priced NDCs in a week. At origin *t*, "as of *t*−k" means the
group's last priced week at or before *t*−k, the same rule as Step 9.

| Feature | Definition |
|---|---|
| `n_labelers` | number of labelers at *t* |
| `d_labelers_12w`, `d_labelers_26w` | change in `n_labelers` vs as of *t*−12 / *t*−26 |
| `labeler_exit_26w` | labelers present as of *t*−26 and gone at *t* |
| `labeler_entry_26w` | labelers present at *t* and not as of *t*−26 |
| `single_source` | `n_labelers` = 1 |

**Leakage checks:**
- **Provenance:** 279,525 rows, latest week read − origin = 0, **0 violations**.
- **Truncation test:** 300 sampled rows were recomputed from NADAC with every week after *t* deleted. **0
  mismatches.**
- **Missing values:** 977 rows have no week ≤ *t*−26, so their 26-week features are NULL.

**Caveats:**
- A "labeler" is whoever labels the package: a manufacturer, a repackager or a store brand. So
  `n_labelers` ≠ the number of real competitors.
- NADAC only lists surveyed NDCs, so the counts reflect NADAC coverage, not the whole market.

## 1. Does the signal exist? (train split only)

**Share of rows moving ≥ 5% in the next 12 weeks:**

| Generics by labelers at *t* | Rows | Move ≥ 5% | Up ≥ 5% | Down ≥ 5% |
|---|---|---|---|---|
| 1 | 38,635 | 49.7% | 24.7% | 24.9% |
| 2 | 10,087 | 55.8% | 26.0% | 29.8% |
| 3 | 7,922 | 63.7% | 28.9% | 34.8% |
| 4+ | 43,762 | 66.3% | 28.5% | 37.8% |

| Generics | Move ≥ 5% | Up | Down |
|---|---|---|---|
| ≥ 1 labeler **exit** in 26 weeks (11,098) vs none (89,129) | 65.6% vs 57.8% | 29.7% vs 26.5% | 35.9% vs 31.3% |
| ≥ 1 labeler **entry** in 26 weeks (15,427) vs none (84,800) | 65.3% vs 57.5% | 25.7% vs 27.1% | **39.6% vs 30.4%** |

- **Yes, the signal exists for generics.** More labelers means more movement, mostly more decreases. A new labeler
  goes with more decreases (39.6% vs 30.4%), which fits "entry → price competition". An exit raises both
  directions a little.
- **Single-source generics with an exit** move a bit more than those without (53.7% vs 49.7%), and **up**: 27.3%
  vs 24.7%.
- **Brands have almost no variation.** 34,212 of 35,062 brand rows have a single labeler, so the features can't
  say much about brands.

## 2. With vs without: the choice (validation only)

| Fold | Validation ROC AUC without → with | Generic | Log-loss without → with |
|---|---|---|---|
| 1 | 0.9138 → 0.9130 | 0.8053 → 0.8034 | 0.3667 → 0.3659 |
| 2 | 0.8428 → 0.8458 | 0.7394 → 0.7444 | 0.4717 → 0.4694 |
| 3 | 0.8058 → 0.8084 | 0.6974 → 0.7019 | 0.4489 → 0.4469 |
| 4 | 0.7656 → 0.7653 | 0.7224 → 0.7233 | 0.5326 → 0.5328 |
| Mean | **0.8320 → 0.8331** | 0.7411 → 0.7433 | 0.4550 → 0.4538 |

**Kept: the version with competition features.** It has the higher mean validation ROC AUC and also the lower
log-loss. The margin is small (+0.001 AUC), and the two versions swap places by fold, so this is a marginal win,
not a clear one.

## 3. Test results (4 folds pooled)

| Target | Segment | Positive rate | ROC AUC: vol rule / without / **with** | PR AUC: without / **with** | Precision@100: without / **with** | Precision@500: without / **with** |
|---|---|---|---|---|---|---|
| move ≥ 5% | all | 0.287 | 0.778 / 0.815 / **0.816** | 0.585 / 0.589 | 0.66 / 0.63 | 0.730 / 0.744 |
| move ≥ 5% | **generic** | 0.387 | 0.708 / 0.727 / **0.729** | 0.593 / 0.597 | 0.66 / 0.63 | 0.730 / 0.744 |
| move ≥ 5% | brand | 0.043 | 0.488 / 0.845 / **0.847** | 0.268 / 0.270 | 0.51 / 0.51 | 0.426 / 0.432 |
| increase ≥ +5% | generic | 0.229 | 0.740 / 0.749 / 0.751 | 0.469 / 0.473 | 0.51 / **0.39** | 0.508 / 0.498 |
| decrease ≤ −5% | generic | 0.250 | 0.679 / 0.707 / 0.710 | 0.389 / 0.394 | 0.15 / **0.24** | 0.222 / 0.246 |

**How the direction rows are measured:**
- **ROC AUC and PR AUC** compare that direction with rows that didn't move ≥ 5%; the opposite direction is left
  out.
- **Precision@N** is the share of the top N of the full list that moved in that direction.
- **Why the scores are low:** the model ranks "will it move", not which way. So direction precision@N is low by
  construction.

**Generic ROC AUC by fold, without → with:**

| Fold | ROC AUC without → with |
|---|---|
| 1 | 0.734 → 0.738 |
| 2 | 0.708 → 0.711 |
| 3 | 0.724 → 0.726 |
| 4 | 0.759 → 0.762 |

**Generic precision@100 by fold, without → with:**

| Fold | Precision@100 without → with |
|---|---|
| 1 | 0.66 → 0.62 |
| 2 | 0.78 → 0.81 |
| 3 | 0.80 → 0.71 |
| 4 | 0.88 → 0.86 |

**Generic precision@500 by fold, without → with:**

| Fold | Precision@500 without → with |
|---|---|
| 1 | 0.722 → 0.716 |
| 2 | 0.688 → 0.706 |
| 3 | 0.716 → 0.724 |
| 4 | 0.788 → 0.778 |

**How much the model uses the features** (fold-4 model, share of gain):

| Feature | Gain share | Rank of 24 features |
|---|---|---|
| `n_labelers` | 1.45% | 14 |
| `single_source` | 0.24% | 16 |
| `labeler_entry_26w` | 0.06% | — |
| `d_labelers_26w` | 0.03% | — |
| `labeler_exit_26w` | 0.01% | — |
| `d_labelers_12w` | 0.00% | — |

**Reading:**
- **The gain is real but tiny.** Generic ROC AUC improves in **all 4 folds**, but only by +0.001 to +0.005
  (pooled +0.0024). Precision@500 rises by 1.4 points pooled.
- **Precision@100 moves both ways by fold** (−9 to +3 points). 100 rows per fold is too few to read much into
  that.
- **Direction:** decreases gain a little (generic AUC 0.707 → 0.710; precision@100 0.15 → 0.24). Increases get
  slightly worse at the very top (precision@100 0.51 → 0.39) and are flat by AUC. That matches the train signal:
  entries → decreases.
- **Why so small, when the raw signal looked clear:** the model already knew most of it.
  - Groups with many labelers also have many NDCs (`n_ndcs`), price tiers and higher recent volatility
    (`vol_26w`, 49% of the gain in 10b).
  - Labeler entries and exits are rare: an exit in 11% of generic train rows.
  - So the new features mostly repeat what volatility and size already say. The model gives them under 2% of its
    gain.

## 4. Watch list grouped by product family

**Origin 2026-04-22**, fold-4 model with competition features. This is the same origin as in 10b.

**Grouping rule** (as in `alerts.py`): two drug groups are one family if they share the first word of the name, or
their 12-week change is identical to 2 decimals and ≥ 30%.

**Counts:**
- 5,759 drug groups in 1,774 families
- 20.5% of groups moved ≥ 5%
- ungrouped, the top 10 rows were only 6 families
- grouped top 10 families: **9 of 10 had at least one member that moved ≥ 5%**

| # | Family (first members) | Members | P(move) | Labelers | Members moved | Range of changes |
|---|---|---|---|---|---|---|
| 1 | Children's cold-cough liquid / cold-cough elixir / Dimaphen DM … | 84 | 0.87 | 1 | 9 | −4.2% … +23.8% |
| 2 | Ivabradine 7.5 / 5 mg | 2 | 0.83 | 6 | 2 | −9.6% … −6.1% |
| 3 | Sulfacetamide-sulfur wash / cleanser / sulfacetamide susp … | 6 | 0.83 | 1 | 5 | −11.6% … +6.3% |
| 4 | Clindamycin-benzoyl peroxide 1.2-3.75% | 1 | 0.82 | 3 | 1 | −8.8% |
| 5 | Leucovorin 25 / 10 / 15 mg … | 4 | 0.82 | 7 | 3 | −17.8% … −3.7% |
| 6 | Fluphenazine 2.5 / 1 / 10 mg … | 5 | 0.81 | 16 | 4 | −30.9% … +0.1% |
| 7 | Isotretinoin 40 / 30 / 20 mg … | 5 | 0.80 | 8 | 3 | −16.8% … +2.2% |
| 8 | Saxagliptin 5 / 2.5 mg, saxagliptin-metformin … | 4 | 0.80 | 3 | 1 | −7.9% … +4.3% |
| 9 | Ticagrelor 60 / 90 mg | 2 | 0.80 | 12 | **0** | −4.8% … +1.4% |
| 10 | Ropinirole ER 8 / 6 / 2 mg … | 12 | 0.79 | 4 | 4 | −16.6% … +43.4% |

**Caveat:** family 1 has 84 members. Store-brand OTC lines chain together through shared stems such as "FT" and
"GNP", the same known gap as the Step 8b-2 families. It now takes one slot instead of four, which is the point.
But "1 family" here is really a cluster of related OTC cough products, and only 9 of its 84 members moved.

## Conclusion

- **Kept: the version with competition features.** It won on validation by a small margin, and it doesn't hurt
  on test: ROC AUC is up slightly in every fold for generics.
- **The improvement is marginal,** about +0.002 AUC and +1.4 points of generic precision@500. Most of the
  competition signal was already captured by volatility and the number of NDCs.
- **The clearest new information** is "a labeler entered → more likely to fall". It shows up as slightly better
  ranking of decreases.
- **For a real watch list, grouping by family matters more than these features.** Without it, one product family
  took 4 of the top 10 slots.
