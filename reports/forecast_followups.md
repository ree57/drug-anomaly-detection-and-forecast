# Step 10b: regression fix and move classification

Generated with `forecast_followups.py`. All numbers are also in `reports/metrics_followups.csv`.

**Honesty note:** we had already seen the Step 10 test results when we designed these two experiments. They are
follow-ups to an observed failure, so the test set is no longer fully "unseen". To limit the damage:
- every setting (number of trees, alpha) was chosen on the validation slices only
- the experiment designs were fixed before running
- test rows were scored once per fold

Each fold still uses the same folds, the purging, the inner validation slice (the latest 3 origins before the fold)
and no `tgt_has_artefact` rows in training. After the settings are chosen on validation, the model is refitted
with them on the training window that ends right before the fold, so it includes the validation origins.

## A. Regression fix (one attempt)

**Changes from Step 10:**
- no `month`
- a **rolling window of the latest 13 origins** (about 12 months) instead of everything since 2022
- predictions **shrunk**: ŷ × alpha, with alpha ∈ {0, 0.25, 0.5, 0.75, 1} picked by validation MAE. Alpha 0
  means the model chooses "predict no change".

| Fold | Validation slice | Trees | Chosen alpha | Validation MAE: naive → chosen |
|---|---|---|---|---|
| 1 | 2024-07-17 … 2024-09-11 | 472 | **1** | 0.1276 → 0.1056 |
| 2 | 2025-01-01 … 2025-02-26 | 1 | **0** | 0.0612 → 0.0612 |
| 3 | 2025-06-18 … 2025-08-13 | 3 | **0.5** | 0.0399 → 0.0398 |
| 4 | 2025-11-05 … 2025-12-31 | 1 | **0** | 0.0553 → 0.0553 |

**Test skill vs naive (1 − MAE/MAE_naive):**

| Fold | All | Generic | Brand |
|---|---|---|---|
| 1 | **−0.584** | −0.599 | −0.291 |
| 2 | 0.000 (alpha 0 = naive) | 0.000 | 0.000 |
| 3 | −0.002 | −0.001 | −0.005 |
| 4 | 0.000 (alpha 0 = naive) | 0.000 | 0.000 |
| 4 folds | **−0.168** (Step 10: −0.216) | −0.174 (Step 10: −0.215) | −0.069 (Step 10: −0.235) |

Without shrinking (alpha 1), skill would have been −0.168 overall. Folds 2 and 4 alone would have been very
slightly positive (+0.001), but that is noise.

**Reading:**
- **It fails again.** The fix doesn't beat naive in any fold. It loses less overall than Step 10, mostly because
  dropping `month` removes the January/February drop predictions. Shrinking mostly switches the model off
  (alpha 0) in folds 2 and 4.
- **Fold 1 can't be rescued with past-only data.**
  - Its validation slice is Jul–Sep 2024, the volatile period. There the model really is better than naive
    (0.106 vs 0.128), so validation correctly picks alpha 1.
  - Then 2025 turns calm and the same model loses badly.
  - No setting chosen on 2024 data could have known that. It is a regime change, not a tuning error.
- **Conclusion:** for "how much will the price change in 12 weeks", naive is the honest answer on this data. The
  signals that predicted 2022–24 moves don't transfer.

## B. Classification: will the price move ≥ 5% in 12 weeks?

**Setup:**
- **Label:** `moves_5pct` = |target_pct_change| ≥ 5. In test, 28.7% of rows move: 38.7% of generics and 4.3% of
  brands.
- **Model:** a LightGBM binary classifier on the same features minus `month`, with an expanding window. Early
  stopping chose 257 / 98 / 170 / 142 trees for folds 1–4.
- **Baselines:**
  - *base rate*: predict the training share of movers for everyone. It cannot rank.
  - *rank by `vol_26w`*: the drugs that were most volatile in the last 26 weeks move next.

**The metrics in plain language:**
- **ROC AUC:** pick one drug that moved and one that didn't. How often does the model give the mover the higher
  score? 0.5 is a coin flip and 1.0 is perfect.
- **PR AUC (average precision):** the average share of real movers in the list as you go down it, summarised over
  all list lengths. A random list scores the base rate, so compare it with the base rate.
- **Precision@100 / @500:** if an analyst checks the top 100 (or 500) drugs in a fold, what share really moved?
  Random picking gives the base rate.
- **Calibration:** when the model says "60% chance", do about 60% of those drugs move?

| Fold | Segment | Base rate | ROC AUC: vol / **LightGBM** | PR AUC: vol / **LightGBM** | Precision@100: vol / **LightGBM** | Precision@500: vol / **LightGBM** |
|---|---|---|---|---|---|---|
| 1 | all | 0.317 | 0.807 / **0.829** | 0.615 / **0.630** | 0.63 / **0.66** | 0.720 / **0.722** |
| 2 | all | 0.281 | 0.780 / **0.803** | 0.511 / **0.563** | 0.30 / **0.78** | 0.482 / **0.690** |
| 3 | all | 0.297 | 0.752 / **0.802** | 0.516 / **0.587** | 0.23 / **0.82** | 0.418 / **0.722** |
| 4 | all | 0.248 | 0.793 / **0.834** | 0.496 / **0.593** | 0.18 / **0.88** | 0.414 / **0.788** |
| 4 folds | all | 0.287 | 0.778 / **0.815** | 0.530 / **0.585** | 0.26 / **0.66** | 0.456 / **0.730** |
| 4 folds | generic | 0.387 | 0.708 / **0.727** | 0.565 / **0.593** | 0.35 / **0.66** | 0.590 / **0.730** |
| 4 folds | brand | 0.043 | 0.488 / **0.845** | 0.053 / **0.268** | 0.02 / **0.51** | 0.064 / **0.426** |

Per-fold brand and generic rows are in the CSV:
- **Brand ROC AUC** is 0.80–0.85 in every fold.
- **Brand precision@100** is 0.42–0.51 in folds 1–3, but only 0.17 in fold 4, where just 2.2% of brands moved.
- **Generic precision@500** is slightly *worse* than the volatility rule in fold 1 (0.722 vs 0.742).

**Calibration (all 4 folds pooled):**

| Predicted bin | Generic: rows, predicted → actual | Brand: rows, predicted → actual |
|---|---|---|
| 0–0.2 | 16,189: 0.131 → 0.107 | 34,860: 0.037 → 0.026 |
| 0.2–0.4 | 26,039: 0.305 → 0.289 | 1,803: 0.280 → 0.299 |
| 0.4–0.6 | 28,422: 0.499 → 0.465 | 270: 0.466 → 0.385 |
| 0.6–0.8 | 16,718: 0.690 → 0.619 | 58: 0.674 → 0.621 |
| 0.8–1.0 | 2,951: 0.833 → 0.712 | 2: 0.818 → 1.000 |

**By fold, mean predicted vs actual:**

| Fold | Mean predicted | Actual |
|---|---|---|
| 1 | 0.404 | 0.317 |
| 2 | 0.299 | 0.281 |
| 3 | 0.292 | 0.297 |
| 4 | 0.250 | 0.248 |

- **Fold 1 is too high.** It learned from the more volatile 2022–24 period, where 45.6% of rows moved.
- **Folds 2–4 are well calibrated on average,** because their training windows include calm 2025 data.
- **Confident generic predictions are too high:** 0.83 predicted vs 0.71 actual in the top bin.

**Top 10 "most likely to move" at origin 2026-04-22.** This is the first origin of fold 4, chosen before looking.
At this origin 5,759 drugs were scored and 20.5% moved ≥ 5%. **All 10 of the top 10 moved**, and 76% of the
top 100 did.

| Drug | P(move) | vol_26w | Last 12 weeks | Actual change in the next 12 weeks |
|---|---|---|---|---|
| Children's cold-cough liquid (4 store-brand versions, one product family) | 0.834 | 0.084 | +78% | +22.8% each |
| Ivabradine 7.5 mg tablet | 0.826 | 0.091 | −52% | −9.6% |
| Isotretinoin 40 mg capsule | 0.823 | 0.086 | −51% | −10.5% |
| Leucovorin 25 mg tablet | 0.821 | 0.088 | −40% | −5.6% |
| Sulfacetamide-sulfur 9-4% wash and cleanser (2 groups) | 0.820 | 0.153 | +17% | −11.6% each |
| Clindamycin-benzoyl peroxide 1.2-3.75% gel | 0.816 | 0.074 | −43% | −8.8% |

Caveat: the top 10 contains only 6 distinct products, because family members get identical features. An analyst
list should group them, as `alerts.py` does.

**What the classifier relies on (fold 4 model, gain):**

| Feature | Gain share |
|---|---|
| vol_26w | 48.9% |
| is_brand | 12.4% |
| classification | 7.0% |
| last_change_size | 4.7% |
| next_12w_includes_january | 4.3% |
| weeks_since_change | 3.9% |
| chg_52w | 3.7% |

**Reading:**
- **It works.** The classifier adds real, stable information in every fold: ROC AUC 0.80–0.83 overall.
- **For an analyst,** checking its top 100 per fold finds real movers 66–88% of the time, against a 25–32% base
  rate.
- **For generics, it is mostly "recent volatility".** It beats the simple `vol_26w` rule by only a little: ROC
  AUC 0.727 vs 0.708, precision@500 0.73 vs 0.59.
- **For brands, it adds clearly more.** Volatility alone is useless there (AUC 0.49, worse than random in two
  folds). The classifier reaches 0.85, mainly by combining brand status with the January window and the recent
  change pattern.
- **Probabilities are usable but optimistic.** They were too high after the volatile training period (fold 1) and
  at the top end for generics. Recalibrating on recent data would help.

## Overall conclusion

- **Size of the move:** predicting *how much* a NADAC price will move in 12 weeks didn't beat "no change", even
  after removing `month`, using a recent window and letting validation shrink the forecast to zero.
- **Whether it moves:** predicting *whether* a price will move ≥ 5% works. It ranks drugs well above chance and
  above a volatility rule, especially for brands.
- **Recommendation:** for this data, the useful forecasting product is a ranked **"likely to move" watch list**,
  not a point forecast of the new price.
