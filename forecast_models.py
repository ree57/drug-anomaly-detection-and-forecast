"""Step 10: forecasting baselines vs LightGBM on data/tables/forecast_dataset (Step 9).

Target y = ln(price t+12 / price t). Models:
  naive      y_hat = 0
  drift      y_hat = clip(chg_12w, -c, c), c picked on the TRAIN split only
  seasonal   brands whose next 12 weeks include a January: median ln change of the group's past REAL price changes
             effective in a January and published <= t (0 if none); everyone else 0
  lgbm       one global LightGBM, objective L1, early stopping on the latest 3 origins of each fold's training rows

Backtest: 4 rolling-origin folds (expanding window, purged). Final fit: train split (2022-24) -> whole test split.
Rows with tgt_has_artefact are never trained or validated on; test metrics are shown with and without them.
Outputs: reports/metrics_forecast.csv; tables printed for reports/forecast_results.md.
"""
import csv
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

SEED = 42
FOLD_STARTS = {1: "2025-01-01", 2: "2025-06-18", 3: "2025-12-03", 4: "2026-04-22"}
TEST_START = "2025-01-01"
VAL_ORIGINS = 3                     # last 3 origins (12 weeks) of a fold's training rows = validation slice
DRIFT_CAPS = [0.05, 0.10, 0.20, None]
MOVED_PCT = 5                       # "the price really moved": |target_pct_change| >= 5
FLAT_BAND = 0.02                    # same +-2% rule as target_direction
BASE_FEATURES = ["chg_4w", "chg_12w", "chg_26w", "chg_52w", "chg_52w_missing", "vol_26w", "n_changes_26w",
                 "weeks_since_change", "last_change_size", "no_change_seen", "is_brand", "classification",
                 "pricing_unit", "log_price", "n_ndcs", "n_distinct_prices", "next_12w_includes_january", "month"]
MARKET_OPTIONS = ["market_move_4w", "market_share_moved_4w"]
CATEGORICAL = ["classification", "pricing_unit"]
PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=31, min_child_samples=200, colsample_bytree=0.8,
              n_estimators=3000, random_state=SEED, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


df = pd.read_parquet("data/tables/forecast_dataset")
df = df[df.split != "purged"].copy()
for c in ["origin_week", "target_week"]:
    df[c] = pd.to_datetime(df[c])
for c in CATEGORICAL:
    df[c] = df[c].astype("category")
for c in ["chg_52w_missing", "no_change_seen", "is_brand", "next_12w_includes_january"]:
    df[c] = df[c].astype(int)
y = "target_log_change"

# ---------- baselines ----------
df["pred_naive"] = 0.0

train = df[(df.split == "train") & ~df.tgt_has_artefact]
cap_mae = {c: (train[y] - train.chg_12w.clip(-(c or np.inf), c or np.inf)).abs().mean() for c in DRIFT_CAPS}
best_cap = min(cap_mae, key=cap_mae.get)
header("drift cap chosen on the TRAIN split (MAE of clip(chg_12w, -c, c))")
print({str(k): round(v, 5) for k, v in cap_mae.items()}, "-> c =", best_cap)
df["pred_drift"] = df.chg_12w.clip(-(best_cap or np.inf), best_cap or np.inf)

# seasonal-naive: past January changes of the same group, published <= t (real events only)
ev = pd.read_parquet("data/tables/price_events",
                     columns=["drug_group", "published_week", "is_january", "mix_only", "artefact_suspect",
                              "old_price", "new_price"])
ev = ev[ev.is_january & ~ev.mix_only & ~ev.artefact_suspect].copy()
ev["published_week"] = pd.to_datetime(ev.published_week)
ev["dl"] = np.log(ev.new_price / ev.old_price)
need = df.loc[(df.is_brand == 1) & (df.next_12w_includes_january == 1), ["drug_group", "origin_week"]].drop_duplicates()
m = need.merge(ev, on="drug_group")
m = m[m.published_week <= m.origin_week]                       # past-only
header("seasonal-naive provenance: latest January event used vs origin")
print(f"pairs: {len(m):,}; max(published_week - origin_week) = {(m.published_week - m.origin_week).max()}; "
      f"violations: {(m.published_week > m.origin_week).sum()}")
seas = m.groupby(["drug_group", "origin_week"]).dl.median().rename("pred_seasonal").reset_index()
df = df.merge(seas, on=["drug_group", "origin_week"], how="left")
mask = (df.is_brand == 1) & (df.next_12w_includes_january == 1)
print(f"brand rows with a January ahead: {mask.sum():,}; with a past January change to copy: "
      f"{df.loc[mask, 'pred_seasonal'].notna().sum():,}")
df["pred_seasonal"] = np.where(mask, df.pred_seasonal.fillna(0.0), 0.0)


# ---------- LightGBM ----------
def fit_predict(test_start, test_mask, features):
    """Fit on rows whose target ends before test_start (purged), early-stop on the latest VAL_ORIGINS origins."""
    pool = df[(df.target_week < pd.Timestamp(test_start)) & ~df.tgt_has_artefact]
    val_origins = sorted(pool.origin_week.unique())[-VAL_ORIGINS:]
    val = pool[pool.origin_week.isin(val_origins)]
    fit = pool[pool.target_week < val_origins[0]]               # inner purge: fit targets end before validation
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(fit[features], fit[y], eval_X=(val[features],), eval_y=(val[y],), eval_metric="l1",
              categorical_feature=CATEGORICAL, callbacks=[lgb.early_stopping(100, verbose=False)])
    info = dict(fit_rows=len(fit), fit_last_origin=fit.origin_week.max().date(),
                fit_last_target=fit.target_week.max().date(), val_origins=f"{val_origins[0].date()}..{val_origins[-1].date()}",
                val_rows=len(val), best_iter=model.best_iteration_, val_mae=model.best_score_["valid_0"]["l1"],
                val_mae_naive=val[y].abs().mean())
    return model, model.predict(df.loc[test_mask, features], num_iteration=model.best_iteration_), info


header("market feature choice: validation MAE per fold (test NOT used for the choice)")
runs, val_rows = {}, []
for mk in MARKET_OPTIONS:
    feats = BASE_FEATURES + [mk]
    for k, s in FOLD_STARTS.items():
        model, pred, info = fit_predict(s, df.fold == k, feats)
        runs[(mk, k)] = (model, pred, info)
        val_rows.append({"market_feature": mk, "fold": k, **info})
val_df = pd.DataFrame(val_rows)
print(val_df.to_string(index=False))
choice = val_df.groupby("market_feature").val_mae.mean()
print(choice.round(6).to_string())
best_mk = choice.idxmin()
print(f"-> keep {best_mk}")

for k in FOLD_STARTS:
    df.loc[df.fold == k, "pred_lgbm"] = runs[(best_mk, k)][1]
    other = [o for o in MARKET_OPTIONS if o != best_mk][0]
    df.loc[df.fold == k, "pred_lgbm_other_market"] = runs[(other, k)][1]

final_feats = BASE_FEATURES + [best_mk]
final_model, final_pred, final_info = fit_predict(TEST_START, df.split == "test", final_feats)
df.loc[df.split == "test", "pred_lgbm_final"] = final_pred
header("final fit (train split -> whole test split)")
print(final_info)

# ---------- metrics ----------
MODELS = {"naive": "pred_naive", "drift": "pred_drift", "seasonal": "pred_seasonal", "lgbm": "pred_lgbm"}


def direction(pred):
    p = np.exp(pred) - 1
    return np.where(p > FLAT_BAND, "up", np.where(p < -FLAT_BAND, "down", "flat"))


def metrics(sub, models):
    out = []
    moved = sub.target_pct_change.abs() >= MOVED_PCT
    naive_mae = sub[y].abs().mean()
    for name, col in models.items():
        err = (sub[y] - sub[col]).abs()
        out.append({"model": name, "rows": len(sub), "mae": err.mean(), "skill": 1 - err.mean() / naive_mae,
                    "dir_acc": (direction(sub[col]) == sub.target_direction).mean(),
                    "rows_moved": int(moved.sum()), "mae_moved": err[moved].mean()})
    return out


rows = []
SETS = [(f"fold {k}", df.fold == k, MODELS) for k in FOLD_STARTS] + \
       [("backtest all folds", df.fold.notna(), MODELS),
        ("final fit, test", df.split == "test", {**MODELS, "lgbm": "pred_lgbm_final"})]
for set_name, set_mask, models in SETS:
    for seg, seg_mask in [("all", True), ("generic", df.is_brand == 0), ("brand", df.is_brand == 1)]:
        for rows_kind, keep in [("all rows", True), ("no artefact", ~df.tgt_has_artefact)]:
            sub = df[set_mask & seg_mask & keep]
            for r in metrics(sub, models):
                rows.append({"set": set_name, "segment": seg, "rows_kind": rows_kind, **r})
res = pd.DataFrame(rows)

header("metrics, ALL test rows (MAE in log units; skill = 1 - MAE/MAE_naive)")
show = res[res.rows_kind == "all rows"].pivot_table(index=["set", "segment"], columns="model",
                                                    values=["mae", "skill", "dir_acc", "mae_moved"], sort=False)
print(show.round(4).to_string())
header("metrics WITHOUT tgt_has_artefact rows (lgbm only, vs all rows)")
print(res[res.model == "lgbm"].pivot_table(index=["set", "segment"], columns="rows_kind",
                                           values=["mae", "skill"], sort=False).round(4).to_string())

header("market feature on TEST (for information only; the choice above used validation)")
for seg, seg_mask in [("all", True), ("generic", df.is_brand == 0), ("brand", df.is_brand == 1)]:
    sub = df[df.fold.notna() & seg_mask]
    print(f"{seg:8s} {best_mk}: {(sub[y] - sub.pred_lgbm).abs().mean():.5f}   "
          f"other: {(sub[y] - sub.pred_lgbm_other_market).abs().mean():.5f}")

header("final model: feature importance (gain), top 10")
gain = pd.Series(final_model.booster_.feature_importance("gain"), index=final_feats).sort_values(ascending=False)
splits = pd.Series(final_model.booster_.feature_importance("split"), index=final_feats)
imp = pd.DataFrame({"gain_share": (gain / gain.sum()).round(4), "splits": splits[gain.index]})
print(imp.head(10).to_string())

header("final model by test origin: bias (mean prediction vs mean target) and MAE")
t = df[df.split == "test"]
by_origin = t.groupby("origin_week").apply(lambda s: pd.Series({
    "rows": len(s), "mean_target": s[y].mean(), "mean_pred": s.pred_lgbm_final.mean(),
    "mae_naive": s[y].abs().mean(), "mae_lgbm": (s[y] - s.pred_lgbm_final).abs().mean()}), include_groups=False)
print(by_origin.round(4).to_string())
header("training period: mean / median target by origin (the market-wide swings the model learned from)")
tr = df[df.split == "train"]
print(tr.groupby("origin_week")[y].agg(["mean", "median"]).round(4).T.to_string())

header("DIAGNOSTIC ONLY (not used to choose a model): final fit WITHOUT month")
no_month = [f for f in final_feats if f != "month"]
_, p_nm, info_nm = fit_predict(TEST_START, df.split == "test", no_month)
t = t.assign(p_nm=p_nm)
print({k: info_nm[k] for k in ["best_iter", "val_mae", "val_mae_naive"]})
for seg, sm in [("all", t.is_brand >= 0), ("generic", t.is_brand == 0), ("brand", t.is_brand == 1)]:
    sub = t[sm]
    print(f"{seg:8s} naive {sub[y].abs().mean():.4f}  with month {(sub[y] - sub.pred_lgbm_final).abs().mean():.4f}"
          f"  without month {(sub[y] - sub.p_nm).abs().mean():.4f}")
jan_feb = t.origin_week.dt.month.isin([1, 2])
print(f"Jan/Feb origins: naive {t[jan_feb][y].abs().mean():.4f}  with month "
      f"{(t[jan_feb][y] - t[jan_feb].pred_lgbm_final).abs().mean():.4f}  without month {(t[jan_feb][y] - t[jan_feb].p_nm).abs().mean():.4f}")

header("final model: what it predicts (share of test rows with |prediction| < 0.005, i.e. ~no change)")
print(t.groupby("is_brand").pred_lgbm_final.apply(lambda s: (s.abs() < 0.005).mean()).round(3).to_string())

header("3 example forecasts (final model; rule: the row closest to the median outcome of its group)")
show_cols = ["drug_group", "origin_week", "chg_12w", "n_changes_26w", "weeks_since_change", "next_12w_includes_january",
             "vol_26w", "pred_naive", "pred_drift", "pred_seasonal", "pred_lgbm_final", y, "target_pct_change"]


def closest_to_median(sub):
    return sub.loc[[(sub[y] - sub[y].median()).abs().sort_values(kind="stable").index[0]]]


ex = pd.concat([
    closest_to_median(t[(t.is_brand == 0) & (t.target_pct_change <= -20)]).assign(example="generic that dropped >= 20%"),
    closest_to_median(t[(t.is_brand == 1) & (t.next_12w_includes_january == 1) & (t.target_direction == "up")])
        .assign(example="brand, January ahead, went up"),
    t[(t.n_changes_26w == 0) & (t[y] == 0)].sort_values(["n_ndcs", "drug_group"], ascending=[False, True]).head(1)
        .assign(example="stable: no change in 26 weeks, none after"),
])
print(ex[["example"] + show_cols].T.to_string())

os.makedirs("reports", exist_ok=True)
res.to_csv("reports/metrics_forecast.csv", index=False, float_format="%.6f", quoting=csv.QUOTE_MINIMAL)
print("\nsaved reports/metrics_forecast.csv")
