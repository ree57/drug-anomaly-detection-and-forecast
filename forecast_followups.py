"""Step 10b: two follow-up experiments after Step 10 (test results already seen -> honest follow-ups).
Every setting is chosen on VALIDATION slices only; test rows are scored once with the chosen settings.

Per fold (same 4 folds and purging as Step 10):
  pool  = rows whose target week ends before the fold starts, without tgt_has_artefact rows
  val   = the latest 3 origins of the pool
  fit   = pool rows whose target ends before the first validation origin (inner purge)
  refit = after choosing settings on val, refit with the chosen number of trees on the training window that ends
          right before the fold (includes the val origins) and predict the fold's test rows

A. regression fix: no `month`; ROLLING window = most recent 13 origins (~12 months) for fit and refit;
   shrink y_hat * alpha, alpha in {0, .25, .5, .75, 1} chosen by validation MAE (alpha 0 = predict no change).
B. classification: label moves_5pct = |target_pct_change| >= 5; LightGBM binary, no `month`, expanding window.
   Baselines: base rate (constant) and ranking by vol_26w.
Outputs: reports/metrics_followups.csv (tables are printed for reports/forecast_followups.md).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
FOLD_STARTS = {1: "2025-01-01", 2: "2025-06-18", 3: "2025-12-03", 4: "2026-04-22"}
VAL_ORIGINS = 3
WINDOW_ORIGINS = 13
ALPHAS = [0, 0.25, 0.5, 0.75, 1]
MOVE_PCT = 5
EXAMPLE_ORIGIN = "2026-04-22"        # first origin of fold 4, fixed before looking at the ranking
FEATURES = ["chg_4w", "chg_12w", "chg_26w", "chg_52w", "chg_52w_missing", "vol_26w", "n_changes_26w",
            "weeks_since_change", "last_change_size", "no_change_seen", "is_brand", "classification",
            "pricing_unit", "log_price", "n_ndcs", "n_distinct_prices", "next_12w_includes_january",
            "market_move_4w"]                                   # Step 10 features minus `month`
CATEGORICAL = ["classification", "pricing_unit"]
COMMON = dict(learning_rate=0.05, num_leaves=31, min_child_samples=200, colsample_bytree=0.8, random_state=SEED,
              deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)

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
df["moves_5pct"] = (df.target_pct_change.abs() >= MOVE_PCT).astype(int)
SEGMENTS = [("all", lambda d: d.is_brand >= 0), ("generic", lambda d: d.is_brand == 0),
            ("brand", lambda d: d.is_brand == 1)]


def last_origins(rows, n):
    return rows[rows.origin_week.isin(sorted(rows.origin_week.unique())[-n:])] if n else rows


def split_fold(start, window):
    pool = df[(df.target_week < pd.Timestamp(start)) & ~df.tgt_has_artefact]
    val_origins = sorted(pool.origin_week.unique())[-VAL_ORIGINS:]
    val = pool[pool.origin_week.isin(val_origins)]
    fit = last_origins(pool[pool.target_week < val_origins[0]], window)
    refit = last_origins(pool, window)
    return fit, val, refit


def describe(rows):
    return f"{rows.origin_week.min().date()}..{rows.origin_week.max().date()} ({rows.origin_week.nunique()} origins, {len(rows):,} rows)"


# ---------------- A. regression fix ----------------
header("A. regression: settings chosen on validation (rolling 13-origin window, no month)")
choice_rows, alpha_rows = [], []
for k, s in FOLD_STARTS.items():
    fit, val, refit = split_fold(s, WINDOW_ORIGINS)
    m = lgb.LGBMRegressor(objective="l1", n_estimators=3000, **COMMON)
    m.fit(fit[FEATURES], fit[y], eval_X=(val[FEATURES],), eval_y=(val[y],), eval_metric="l1",
          categorical_feature=CATEGORICAL, callbacks=[lgb.early_stopping(100, verbose=False)])
    p_val = m.predict(val[FEATURES], num_iteration=m.best_iteration_)
    val_mae = {a: (val[y] - a * p_val).abs().mean() for a in ALPHAS}
    alpha = min(val_mae, key=val_mae.get)                      # ties -> smallest alpha (listed first)
    for a, v in val_mae.items():
        alpha_rows.append({"fold": k, "alpha": a, "val_mae": v})
    final = lgb.LGBMRegressor(objective="l1", n_estimators=m.best_iteration_, **COMMON)
    final.fit(refit[FEATURES], refit[y], categorical_feature=CATEGORICAL)
    test = df.fold == k
    df.loc[test, "pred_raw"] = final.predict(df.loc[test, FEATURES])
    df.loc[test, "pred_shrunk"] = alpha * df.loc[test, "pred_raw"]
    choice_rows.append({"fold": k, "fit": describe(fit), "val": describe(val), "refit": describe(refit),
                        "best_iter": m.best_iteration_, "alpha": alpha,
                        "val_mae_naive": val[y].abs().mean(), "val_mae_chosen": val_mae[alpha]})
print(pd.DataFrame(choice_rows).to_string(index=False))
print("\nvalidation MAE by alpha:")
print(pd.DataFrame(alpha_rows).pivot(index="alpha", columns="fold", values="val_mae").round(5).to_string())

reg_rows = []
for k in list(FOLD_STARTS) + ["all"]:
    sel = df.fold.notna() if k == "all" else df.fold == k
    for seg, segf in SEGMENTS:
        sub = df[sel & segf(df)]
        naive = sub[y].abs().mean()
        for name, col in [("raw (alpha 1)", "pred_raw"), ("shrunk (chosen alpha)", "pred_shrunk")]:
            mae = (sub[y] - sub[col]).abs().mean()
            reg_rows.append({"part": "A", "fold": k, "segment": seg, "model": name, "rows": len(sub),
                             "mae_naive": naive, "mae": mae, "skill": 1 - mae / naive})
reg = pd.DataFrame(reg_rows)
header("A. test results: skill vs naive (1 - MAE/MAE_naive), per fold and segment")
print(reg.pivot_table(index=["fold", "segment"], columns="model", values=["mae_naive", "mae", "skill"], sort=False)
         .round(4).to_string())

# ---------------- B. classification ----------------
header("B. classification: settings chosen on validation (expanding window, no month)")
cls_choice = []
for k, s in FOLD_STARTS.items():
    fit, val, refit = split_fold(s, None)
    m = lgb.LGBMClassifier(objective="binary", n_estimators=3000, **COMMON)
    m.fit(fit[FEATURES], fit.moves_5pct, eval_X=(val[FEATURES],), eval_y=(val.moves_5pct,), eval_metric="binary_logloss",
          categorical_feature=CATEGORICAL, callbacks=[lgb.early_stopping(100, verbose=False)])
    final = lgb.LGBMClassifier(objective="binary", n_estimators=m.best_iteration_, **COMMON)
    final.fit(refit[FEATURES], refit.moves_5pct, categorical_feature=CATEGORICAL)
    test = df.fold == k
    df.loc[test, "p_move"] = final.predict_proba(df.loc[test, FEATURES])[:, 1]
    df.loc[test, "p_base"] = refit.moves_5pct.mean()          # base-rate baseline: training share that moved
    cls_choice.append({"fold": k, "fit": describe(fit), "val": describe(val), "refit": describe(refit),
                       "best_iter": m.best_iteration_, "val_auc": roc_auc_score(val.moves_5pct, m.predict_proba(val[FEATURES])[:, 1]),
                       "train_base_rate": refit.moves_5pct.mean()})
    if k == 4:
        model_fold4 = final
print(pd.DataFrame(cls_choice).round(4).to_string(index=False))
df["score_vol"] = df.vol_26w.fillna(0)


def prec_at(sub, col, n):
    return sub.sort_values(col, ascending=False, kind="stable").head(n).moves_5pct.mean()


cls_rows = []
for k in list(FOLD_STARTS) + ["all"]:
    sel = df.fold.notna() if k == "all" else df.fold == k
    for seg, segf in SEGMENTS:
        sub = df[sel & segf(df)]
        for name, col in [("base rate", "p_base"), ("rank by vol_26w", "score_vol"), ("lgbm classifier", "p_move")]:
            const = name == "base rate"   # a constant guess: no ranking, by definition
            cls_rows.append({"part": "B", "fold": k, "segment": seg, "model": name, "rows": len(sub),
                             "actual_rate": sub.moves_5pct.mean(),
                             "roc_auc": 0.5 if const else roc_auc_score(sub.moves_5pct, sub[col]),
                             "pr_auc": sub.moves_5pct.mean() if const else average_precision_score(sub.moves_5pct, sub[col]),
                             "prec_at_100": np.nan if const else prec_at(sub, col, 100),
                             "prec_at_500": np.nan if const else prec_at(sub, col, 500)})
cls = pd.DataFrame(cls_rows)
header("B. test results (constant base rate: ROC AUC 0.5 and PR AUC = actual rate by definition)")
print(cls.pivot_table(index=["fold", "segment"], columns="model", values=["actual_rate", "roc_auc", "pr_auc", "prec_at_100",
                      "prec_at_500"], sort=False).round(3).to_string())

header("B. calibration: predicted vs actual share that moved, 5 fixed-width bins, all 4 folds pooled")
bt = df[df.fold.notna()].copy()
bt["bin"] = pd.cut(bt.p_move, [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
cal = bt.groupby(["is_brand", "bin"], observed=False).agg(rows=("p_move", "size"), mean_predicted=("p_move", "mean"),
                                                          actual_rate=("moves_5pct", "mean")).round(3)
print(cal.to_string())
print(f"overall: mean predicted {bt.p_move.mean():.3f}, actual {bt.moves_5pct.mean():.3f}")
by_fold = bt.groupby("fold").agg(mean_predicted=("p_move", "mean"), actual_rate=("moves_5pct", "mean"),
                                 train_base_rate=("p_base", "first")).round(3)
print(by_fold.to_string())

header(f"B. top 10 'most likely to move' at origin {EXAMPLE_ORIGIN} (fold 4 model) and what happened")
ex = df[df.origin_week == pd.Timestamp(EXAMPLE_ORIGIN)].sort_values("p_move", ascending=False, kind="stable")
print(f"rows at this origin: {len(ex):,}; share that moved >= 5%: {ex.moves_5pct.mean():.3f}; "
      f"top 10 that moved: {ex.head(10).moves_5pct.sum()}; top 100: {ex.head(100).moves_5pct.mean():.2f}")
cols = ["drug_group", "p_move", "vol_26w", "n_changes_26w", "chg_12w", "weeks_since_change", "target_pct_change", "moves_5pct"]
print(ex.head(10)[cols].round(4).to_string(index=False))

header("B. classifier feature importance (gain), fold 4 model, top 10")
g = pd.Series(model_fold4.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
print((g / g.sum()).round(4).head(10).to_string())

pd.concat([reg, cls]).to_csv("reports/metrics_followups.csv", index=False, float_format="%.6f")
print("\nsaved reports/metrics_followups.csv")
