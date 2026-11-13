# Step 10: forecasting baselines vs LightGBM

Generated with `forecast_models.py`. All metrics are also in `reports/metrics_forecast.csv`. The target is
y = ln(price *t*+12 / price *t*), and MAE is in log units (0.046 ≈ 4.6%).

**Bottom line:** no model beats "predict no change" (naive) on the 2025–26 test period, overall or for
generics. LightGBM is clearly worse in the first fold and in the final fit, and only ties naive in folds 2–4.
The one place another model helps is **brands that really moved**: seasonal-naive has a lower MAE there and better
direction accuracy for brands.

## Models

| Model | Prediction |
|---|---|
| naive | 0 |
| drift | clip(`chg_12w`, −c, c). Cap chosen on the train split: **c = 0.05**. Train MAE was 0.0987, rising to 0.110 / 0.123 / 0.139 for c = 0.10 / 0.20 / none. |
| seasonal | For brands with a January in the next 12 weeks: median ln change of the group's past **real** January changes published ≤ *t*. 0 for everyone else. 10,930 of 16,200 such brand rows had a past January to copy. Provenance: 19,961 (row, event) pairs, 0 events after the origin. |
| LightGBM | Global model, `objective=l1`, learning rate 0.05, 31 leaves, min 200 rows per leaf, 80% features per tree, early stopping (100) on the latest 3 origins of each fold's training rows, inner purge, `random_state=42`, `deterministic=True`. Rows with `tgt_has_artefact` left out of fitting and validation. |

## Results

| Set | Segment | MAE naive | drift | seasonal | **LightGBM** | Skill LightGBM | Dir. acc naive / seasonal / LightGBM | MAE moved ≥5%: naive / seasonal / LightGBM |
|---|---|---|---|---|---|---|---|---|
| fold 1 | all | 0.0485 | 0.0623 | 0.0494 | 0.0852 | **−0.755** | 0.496 / 0.497 / 0.456 | 0.126 / 0.125 / 0.181 |
| fold 2 | all | 0.0460 | 0.0517 | 0.0462 | 0.0460 | −0.000 | 0.506 / 0.513 / 0.506 | 0.130 / 0.129 / 0.130 |
| fold 3 | all | 0.0496 | 0.0582 | 0.0502 | 0.0497 | −0.001 | 0.471 / 0.485 / 0.471 | 0.133 / 0.131 / 0.133 |
| fold 4 | all | 0.0409 | 0.0470 | 0.0409 | 0.0409 | −0.000 | 0.547 / 0.547 / 0.547 | 0.127 / 0.127 / 0.127 |
| backtest (4 folds) | all | 0.0464 | 0.0550 | 0.0468 | 0.0564 | −0.216 | 0.505 / 0.510 / 0.494 | 0.129 / 0.128 / 0.145 |
| | generic | 0.0614 | 0.0710 | 0.0614 | 0.0746 | −0.215 | 0.346 / 0.346 / 0.348 | 0.128 / 0.128 / 0.146 |
| | brand | 0.0097 | 0.0158 | 0.0112 | 0.0120 | −0.235 | 0.893 / **0.912** / 0.850 | 0.147 / **0.126** / 0.144 |
| final fit (train → whole test) | all | 0.0464 | 0.0550 | 0.0468 | 0.0627 | −0.351 | 0.505 / 0.510 / 0.512 | 0.129 / 0.128 / 0.147 |
| | generic | 0.0614 | 0.0710 | 0.0614 | 0.0822 | −0.339 | 0.346 / 0.346 / 0.385 | 0.128 / 0.128 / 0.148 |
| | brand | 0.0097 | 0.0158 | 0.0112 | 0.0149 | −0.538 | 0.893 / 0.912 / 0.824 | 0.147 / 0.126 / 0.133 |

**Notes on the table:**
- **Drift** is worse than naive everywhere: skill −0.19 overall, −0.63 for brands. Its only plus is generic
  direction accuracy, 0.39 vs 0.35 for naive.
- **Rows without `tgt_has_artefact`:** LightGBM's MAE changes by at most 0.0012. Backtest: 0.0556 vs 0.0564.
  Final fit: 0.0619 vs 0.0627. The conclusions don't change.
- **Per-fold, per-segment detail** is in the CSV. The only positive LightGBM skill anywhere is fold 3 brands, at
  +0.002, which is negligible.

## Market feature

- **The choice:** I chose on validation, averaged over the 4 folds. `market_move_4w` had MAE 0.06568 and
  `market_share_moved_4w` had 0.06631, so I kept `market_move_4w`.
- **The catch:** the difference is about 1%, which is within noise. On test, shown for information only and not
  used for the choice, the share-moved version did better: 0.0506 vs 0.0564 backtest MAE. Both are still worse
  than naive (0.0464).
- **Verdict:** the validation slices can't really tell the two features apart.

## Why LightGBM doesn't beat naive

**1. The training period had market-wide swings that 2025–26 didn't.**

| | Mean 12-week change after the origin |
|---|---|
| 2022–23 origins | about −3% to +1% |
| Origins 2024-01-31 / 2024-02-28 | −10.4% / −12.2% |
| Origins 2024-07-17 / 2024-08-14 / 2024-09-11 | +10.4% / +9.7% / −10.0% |

- **What the model learned:** "a January/February origin is followed by a big drop". `month` has 9.6% of the
  gain. It wasn't learned on brands; it is one year's market event.
- **What happened in test:** the model predicted −11% on average for the 2025-01-29 and 2025-02-26 origins and
  about −9% for 2026-01-28 and 2026-02-25. The truth was +4.3%, +2.0%, −1.5% and −0.7%. The two 2025
  origins are in fold 1 and have its worst errors: MAE 0.159 and 0.143 vs naive 0.066 and 0.049. The 2026 pair
  costs 0.095 and 0.090 vs naive 0.048 and 0.044 in the final fit.
- **Diagnostic only, not used to choose:** the same final model without `month` scores MAE 0.0528. That's better
  than 0.0627, but still worse than naive's 0.0464. On Jan/Feb origins the numbers are 0.069 (without month) vs
  0.113 (with month) vs 0.055 (naive).
- **Open question:** I haven't investigated what caused the 2024 swings.

**2. The test period is calm.** Generics move about half as much in 2025–26: naive MAE is 0.061, against 0.110
in train. A model fitted to 2022–24 expects bigger moves than 2025–26 delivers.

**3. Early stopping did its job in folds 2–4.**
- **What happened:** those folds validate on 2025 origins, where nothing the model learned helps. Early stopping
  kept **1 tree** (validation MAE ≈ naive, e.g. fold 2: 0.0616 vs 0.0612), so the model is essentially "predict
  the median", about 0.
- **Why fold 1 differs:** its validation slice is Jul–Sep 2024, which is volatile. There the model *did* beat
  naive: 0.106 vs 0.128. So it kept 397 trees, and those don't carry over to 2025.

**4. Brand drift changed.**
- **In training:** 48% of brand rows had exactly zero change.
- **In test:** only 20% did, as brands now drift by tiny amounts while staying mostly within ±2%.
- **Effect:** naive is very hard to beat for brands (MAE 0.0097). Any learned January bump costs more on the many
  brands that don't move than it gains on the few that do.

## Feature importance (final model, gain)

| Feature | Gain share |
|---|---|
| weeks_since_change | 12.8% |
| chg_52w | 11.8% |
| last_change_size | 10.7% |
| month | 9.6% |
| chg_12w | 9.1% |
| chg_4w | 6.6% |
| log_price | 6.2% |
| vol_26w | 5.9% |
| chg_26w | 5.8% |
| n_changes_26w | 5.1% |

**Reading it:**
- **Price history dominates.** The model relies mostly on the drug's own recent history: how long since the last
  real change, the trend over a year, and the size of the last change. These are momentum and mean-reversion
  patterns.
- **Calendar is the fourth signal.** `month` is important, and per the diagnosis above that's a problem, not a
  strength.
- **Little else.** Price level, volatility and the number of changes add a bit. Next comes
  `next_12w_includes_january` at 4.6%. Brand/generic (2.8%) and the market feature (2.8%) contribute little.
- **What gain can't tell you:** it doesn't say which direction a feature pushes, and correlated features (the four
  `chg_*`) share credit.

## Example forecasts (final model)

Each example is the test row closest to the median outcome of its group:

| | Diltiazem 24h ER 180 mg (generic), origin 2025-11-05 | Kerendia 10 mg (brand, January ahead), 2025-11-05 | Rocuronium 50 mg/5 ml (generic, stable), 2025-05-21 |
|---|---|---|---|
| Last 12 weeks | −5.7% | 0.0% | 0.0% (no change for 30 weeks) |
| Naive | 0 | 0 | **0** |
| Drift | −0.05 | 0.000 | **0** |
| Seasonal | 0 | **+0.049** | **0** |
| LightGBM | +0.024 | +0.029 | −0.008 |
| **Truth** | **−0.294 (−25.5%)** | **+0.041 (+4.2%)** | **0 (0%)** |

- **Diltiazem:** nobody predicted the 25% drop. LightGBM even predicted a small rise (mean reversion after a
  12-week fall).
- **Kerendia:** this is where history helps. Seasonal (its past January change) is closest, and LightGBM is
  better than naive.
- **Rocuronium:** naive is perfect. LightGBM adds a small −0.8% error.

## What this means for Step 11+ (not done here)

These are possible next directions:
- Drop calendar features learned from a single year, or train on a recent rolling window.
- Shrink predictions toward 0 unless the model is confident.
- Treat brands separately, with a January rule plus naive.
- Forecast "will it move ≥ 5%?" as a classification, which may be more useful than a point forecast.
- Understand the 2024 swings before trusting any model trained on them.
