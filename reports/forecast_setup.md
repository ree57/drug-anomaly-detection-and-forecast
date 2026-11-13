# Step 9: forecasting problem setup (no model yet)

Built by `forecast_setup.py` from `sql/forecast_dataset.sql` into `data/tables/forecast_dataset/`.
Every number below was printed by that script.

## The problem

| Term | Definition |
|---|---|
| Forecast origin *t* | The published week we forecast from. Only weeks ≤ *t* may be used. Origins are every 4th published week from 2022-01-05 (wk % 4 = 0). |
| Horizon | 12 published weeks (84 days). |
| Target | `target_log_change` = ln(price at *t*+12 / price at *t*) on the group's published median. Also `target_pct_change` and `target_direction`: up / down / flat (within ±2%). |
| Row | One per (drug group, origin), if the group is priced at *t* and has ≥ 26 priced weeks at or before *t*. |

Why every 4 weeks: neighbouring weeks share most of their 12-week window, so weekly origins would mostly copy
rows. Why a time split: a random split lets the model learn from rows next to the test rows in time.
**Purge:** a training row is used only if its whole target window ends before the test period (or fold) starts.

## Rows

- **Total:** 302,533 candidate rows.
- **Kept:** 279,525 rows with a known *t*+12 price.
- **Dropped, 23,008 rows:**
  - **17,394** "beyond data end": the last 3 origins (2026-09-09 … 2026-11-04) have no *t*+12 week yet.
  - **5,614** "not priced at *t*+12": the group had a gap that week. That's 2.0% of rows with an origin
    old enough (2,511 train, 152 purged, 2,951 test).

**Artefacts and mix-only weeks:**
- **Price series:** features and target use the raw published median. That's what NADAC shows, and week *t*
  can't be "cleaned" without later weeks.
- **Count-type features:** `n_changes_26w`, `weeks_since_change` and `last_change_size` count only real changes
  (not `mix_only`, not `artefact_suspect`). Both flags are known when a week is published.
- **Target window:** the flags `tgt_has_artefact` and `tgt_has_mix_only` mark an artefact or mix-only event in
  weeks *t*+1 … *t*+12. They look into the future, so they are never features.
  - `tgt_has_artefact` covers 195 train rows and 142 test rows. Proposal for Step 10: train without these rows
    and report test metrics both with and without them.
  - `tgt_has_mix_only` covers 361 train rows and 259 test rows. These are kept.

## Splits (purged)

| Split | Origins | Rows | Groups | Last target week |
|---|---|---|---|---|
| train | 2022-07-20 … 2024-09-11 (29) | 135,468 | 5,513 | 2024-12-04 |
| purged (not used) | 2024-10-09, 2024-11-06, 2024-12-04 (3) | 16,745 | 5,698 | 2025-02-26 |
| test | 2025-01-01 … 2026-08-12 (22) | 127,312 | — | 2026-11-04 |

I made a mistake in the plan: I listed only 2024-10-09 as purged. In fact all three 2024 origins after
2024-09-11 have targets in 2025.

**Rolling-origin backtest folds for Step 10:**
- Each fold trains on every row whose target week is before the fold start (an expanding window).
- Each fold tests on its own block of origins.

| Fold | Test origins | Test rows | Train rows | Last train origin / target week |
|---|---|---|---|---|
| 1 | 2025-01-01 … 2025-05-21 (6) | 34,807 | 135,468 | 2024-09-11 / 2024-12-04 |
| 2 | 2025-06-18 … 2025-11-05 (6) | 35,104 | 169,510 | 2025-02-26 / 2025-05-21 |
| 3 | 2025-12-03 … 2026-03-25 (5) | 28,708 | 204,612 | 2025-08-13 / 2025-11-05 |
| 4 | 2026-04-22 … 2026-08-12 (5) | 28,693 | 233,611 | 2025-12-31 / 2026-03-25 |

## Features (all from published weeks ≤ *t*)

| Feature | Reason |
|---|---|
| `chg_4w`, `chg_12w`, `chg_26w`, `chg_52w` | ln(p_t / last price at or before *t*−k): recent trend and momentum, or mean reversion. `chg_52w` is NULL for 14.9% of rows (less than a year of history); `chg_52w_missing` flags them. `chg_26w` is NULL for 977 rows. |
| `vol_26w` | Standard deviation of weekly log changes over weeks *t*−25 … *t*: how jumpy the drug is. NULL for 57 rows with a single change in the window. |
| `n_changes_26w` | Real price changes in the last 26 weeks: does it move at all? |
| `weeks_since_change` | Long-flat drugs tend to stay flat. If there was never a real change (1.1% of rows), it counts from the group's first week, and `no_change_seen` is set. |
| `last_change_size` | Log size of the last real change (0 if none). A big recent move can be followed by a correction. |
| `is_brand`, `classification` | Brands (27.5% of rows) follow list prices; generics follow survey averages. |
| `pricing_unit` | EA / ML / GM: the product type. |
| `log_price` | Cheap and expensive drugs move differently in % terms. |
| `n_ndcs`, `n_distinct_prices` | More products and more price tiers give a noisier median. |
| `next_12w_includes_january` | Brand list-price increases cluster in January. It's calendar, so known at *t* (27.4% of rows). |
| `month` | Other seasonality. |
| `market_move_4w` | Median 4-week log change of all generic groups at *t*: whether the generic market as a whole is drifting. |

**Caveat on `market_move_4w`:** it is exactly 0 at 28 of 54 origins and ranges from −0.132 to +0.081. It is weak
but not constant. I also stored `market_share_moved_4w`, the share of generics whose price changed in 4 weeks
(median 0.93, range 0.001–0.98), as a candidate. Step 10 can decide whether to use it.

## Leakage checks (evidence)

1. **Provenance:** every row stores the latest week index any feature read (`feat_max_src_wk`). Over all 279,525
   rows, the maximum of (source week − origin) is **0**, with **0 violations**.
2. **Truncation test:** for 400 random rows from 39 origins, every feature was recomputed in pandas from data with
   all weeks after *t* deleted.
   - **4,400 feature values compared, 0 mismatches.**
   - `market_move_4w` was recomputed for the same 39 origins: **0 mismatches.**
3. **Purge:**
   - **Final split:** the last train target week is 2024-12-04, and 0 train rows have a target week ≥ 2025-01-01.
   - **Folds:** 0 overlapping rows in every fold.

## The target

ln(price *t*+12 / price *t*), percentiles:

| Split | Class | Rows | p01 | p05 | p25 | p50 | p75 | p95 | p99 | Exactly 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| train | generic | 100,406 | −0.539 | −0.301 | −0.080 | −0.001 | 0.056 | 0.234 | 0.436 | 5.2% |
| train | brand | 35,062 | −0.140 | −0.004 | 0.000 | 0.000 | 0.001 | 0.054 | 0.094 | 48.0% |
| test | generic | 90,319 | −0.311 | −0.155 | −0.040 | −0.000 | 0.031 | 0.137 | 0.264 | 4.2% |
| test | brand | 36,993 | −0.032 | −0.004 | −0.001 | 0.000 | 0.001 | 0.039 | 0.086 | 19.6% |

**Direction:**

| Split | Class | Down | Flat | Up |
|---|---|---|---|---|
| train | generic | 42.2% | 21.6% | 36.2% |
| train | brand | 1.8% | 82.9% | 15.2% |
| test | generic | 34.9% | 34.5% | 30.5% |
| test | brand | 1.2% | 89.3% | 9.4% |

**Brands and January:**
- **How often:** the next 12 weeks include a January for 27.1% of brand rows in train and 18.1% in test.
- **Train:** with a January, 45.1% of brand rows go up; without one, 4.1%.
- **Test:** with a January, 37.8% go up; without one, 3.2%.

**What changes between periods:**
- **Generics** move about half as much in 2025–26 as in 2022–24.
- **Brands** are flat (within ±2%) most of the time. But exact zero changes dropped from 48% of brand rows to
  20%, so brands now drift by tiny amounts. Any model trained on 2022–24 faces this shift.

## First baseline: predict no change (ŷ = 0)

MAE = mean |target| (log units; 0.046 ≈ 4.6%):

| Set | Rows | MAE all | MAE generic | MAE brand | MAE without `tgt_has_artefact` rows | Median \|y\| |
|---|---|---|---|---|---|---|
| train | 135,468 | 0.0862 | 0.1104 | 0.0167 | 0.0855 | 0.0420 |
| test | 127,312 | 0.0464 | 0.0614 | 0.0097 | 0.0456 | 0.0195 |
| fold 1 | 34,807 | 0.0485 | 0.0657 | 0.0080 | 0.0481 | 0.0206 |
| fold 2 | 35,104 | 0.0460 | 0.0610 | 0.0099 | 0.0448 | 0.0192 |
| fold 3 | 28,708 | 0.0496 | 0.0625 | 0.0175 | 0.0492 | 0.0233 |
| fold 4 | 28,693 | 0.0409 | 0.0557 | 0.0038 | 0.0401 | 0.0155 |

Every model in Step 10 must beat these numbers, per fold and separately for generics and brands.
