"""Step 11: precompute small files for the Streamlit dashboard (dashboard/app.py) into dashboard_data/.

  groups.parquet     one row per drug group: small integer id (gid), name, unit, class
  alerts.parquet     every high/medium alert (Step 8d detector), all weeks, grouped into families, with the
                     plain-English reason and caution flags (alert_lib.py)
  watchlist.parquet  P(move >= 5% up or down in the next 12 weeks) from the Step 10c classifier (with competition
                     features), grouped into families per origin:
                       - test origins 2025-01-01 .. 2026-08-12: scored walk-forward by the model of their own fold
                         (it never saw their outcome); actual outcome attached
                       - newest origins (no 12-week outcome yet): scored by a "current" model trained the same way on
                         every row whose outcome is known
  prices.parquet     weekly group median price
  events.parquet     every price event with mix-only / artefact flags and its alert confidence
  summary.json       key metrics for the About page, read from the reports/metrics_*.csv files

Pandas + LightGBM only (no Spark). Run after the pipeline: python export_dashboard.py
"""
import json
from datetime import timedelta
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from alert_lib import cautions, load_scored_events, reason, with_families
from families import assign_families, derive_store_prefixes

OUT = "dashboard_data"
SEED = 42
FOLD_STARTS = {1: "2025-01-01", 2: "2025-06-18", 3: "2025-12-03", 4: "2026-04-22"}
VAL_ORIGINS = 3
MOVE_PCT = 5
FEATURES = ["chg_4w", "chg_12w", "chg_26w", "chg_52w", "chg_52w_missing", "vol_26w", "n_changes_26w",
            "weeks_since_change", "last_change_size", "no_change_seen", "is_brand", "classification", "pricing_unit",
            "log_price", "n_ndcs", "n_distinct_prices", "next_12w_includes_january", "market_move_4w",
            "n_labelers", "d_labelers_12w", "d_labelers_26w", "labeler_exit_26w", "labeler_entry_26w", "single_source"]
CATEGORICAL = ["classification", "pricing_unit"]
COMMON = dict(learning_rate=0.05, num_leaves=31, min_child_samples=200, colsample_bytree=0.8, random_state=SEED,
              deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
os.makedirs(OUT, exist_ok=True)

# ---------- groups ----------
gd = pd.read_parquet("data/tables/group_dim")
prefix_info = derive_store_prefixes(gd.description)
prefixes = list(prefix_info)
print("store-brand prefixes:", prefixes)
groups = gd.sort_values("drug_group").reset_index(drop=True)
groups["gid"] = np.arange(len(groups), dtype=np.int32)
groups["is_brand"] = groups.classification != "G"
gid = dict(zip(groups.drug_group, groups.gid))
groups[["gid", "drug_group", "description", "pricing_unit", "classification", "is_brand", "first_seen",
        "last_seen"]].to_parquet(f"{OUT}/groups.parquet", index=False)

# ---------- alerts ----------
ev = with_families(load_scored_events(), prefixes)
al = ev[ev.confidence.isin(["high", "medium"])].copy()
al["reason"] = [reason(r) for _, r in al.iterrows()]
al["caution_text"] = ["; ".join(c.removeprefix("caution: ") for c in cautions(r)) for _, r in al.iterrows()]
fam = al.groupby("family").agg(family_score=("score_final", "max"), family_tier=("tier", "max"),
                               family_size=("drug_group", "size"))
al = al.join(fam, on="family")
al["family_rank"] = al.groupby("published_week").family_score.rank(method="dense", ascending=False).astype(int)
al["period"] = np.where(pd.to_datetime(al.published_week) < "2025-01-01", "2022-24 (detector calibration period)",
                        "2025-26 (out of sample)")
al["gid"] = al.drug_group.map(gid).astype(np.int32)
alert_cols = ["published_week", "family", "family_rank", "family_tier", "family_size", "gid", "description",
              "pricing_unit", "classification", "is_brand", "old_price", "new_price", "pct_change", "confidence",
              "medium_source", "score_final", "reason", "caution_text", "caution_class_changed", "caution_after_gap",
              "gap_weeks_missing", "caution_jan_brand_round", "caution_jan_brand_not_round", "period"]
al[alert_cols].sort_values(["published_week", "family_rank", "score_final"], ascending=[True, True, False]) \
    .to_parquet(f"{OUT}/alerts.parquet", index=False)
print(f"alerts: {len(al):,} events in {al.family.nunique():,} families over {al.published_week.nunique()} weeks")

# ---------- watch list ----------
def load_forecast(path):
    df = pd.read_parquet(path).merge(
        pd.read_parquet("data/tables/forecast_competition").drop(columns=["origin_wk", "comp_max_src_wk"]),
        on=["drug_group", "origin_week"], how="left", validate="one_to_one")
    for c in ["origin_week", "target_week"]:
        df[c] = pd.to_datetime(df[c])
    for c in ["chg_52w_missing", "no_change_seen", "is_brand", "next_12w_includes_january"]:
        df[c] = df[c].astype(int)
    return df


known = load_forecast("data/tables/forecast_dataset")
known = known[known.split != "purged"]
latest = load_forecast("data/tables/forecast_latest")
cats = {c: pd.api.types.union_categoricals([known[c].astype("category"), latest[c].astype("category")]).categories
        for c in CATEGORICAL}
for d in (known, latest):
    for c in CATEGORICAL:
        d[c] = pd.Categorical(d[c], categories=cats[c])
known["moves_5pct"] = (known.target_pct_change.abs() >= MOVE_PCT).astype(int)


def fit_classifier(start):
    """Step 10c procedure: early stopping on the latest 3 origins (inner purge), then refit on everything."""
    pool = known[(known.target_week < pd.Timestamp(start)) & ~known.tgt_has_artefact]
    vo = sorted(pool.origin_week.unique())[-VAL_ORIGINS:]
    fit, val = pool[pool.target_week < vo[0]], pool[pool.origin_week.isin(vo)]
    m = lgb.LGBMClassifier(objective="binary", n_estimators=3000, **COMMON)
    m.fit(fit[FEATURES], fit.moves_5pct, eval_X=(val[FEATURES],), eval_y=(val.moves_5pct,),
          eval_metric="binary_logloss", categorical_feature=CATEGORICAL,
          callbacks=[lgb.early_stopping(100, verbose=False)])
    final = lgb.LGBMClassifier(objective="binary", n_estimators=m.best_iteration_, **COMMON)
    final.fit(pool[FEATURES], pool.moves_5pct, categorical_feature=CATEGORICAL)
    print(f"  model for origins from {start}: trained on {pool.origin_week.min().date()}..{pool.origin_week.max().date()}"
          f" ({len(pool):,} rows), {m.best_iteration_} trees")
    return final


parts = []
for k, s in FOLD_STARTS.items():
    m = fit_classifier(s)
    t = known[known.fold == k].copy()
    t["p_move"] = m.predict_proba(t[FEATURES])[:, 1]
    t["model"] = f"walk-forward (fold {k})"
    parts.append(t)
current_start = (pd.Timestamp(known.target_week.max()).date() + timedelta(days=1)).isoformat()
m = fit_classifier(current_start)
t = latest.copy()
t["p_move"] = m.predict_proba(t[FEATURES])[:, 1]
t["model"] = "current (outcome not known yet)"
parts.append(t)
wl = pd.concat(parts, ignore_index=True)
wl["description"] = wl.drug_group.str.split("|").str[0]
wl["pct_12w_before"] = (np.exp(wl.chg_12w) - 1) * 100
wl = assign_families(wl, "origin_week", "description", "pct_12w_before", prefixes)
wl = wl.join(wl.groupby("family").p_move.max().rename("family_p"), on="family")
wl["family_rank"] = wl.groupby("origin_week").family_p.rank(method="dense", ascending=False).astype(int)
wl["outcome_known"] = wl.target_pct_change.notna()
wl["moved_5pct"] = np.where(wl.outcome_known, wl.target_pct_change.abs() >= MOVE_PCT, np.nan)
wl["gid"] = wl.drug_group.map(gid).astype(np.int32)
wl_cols = ["origin_week", "target_week", "model", "family", "family_rank", "family_p", "gid", "description",
           "is_brand", "p_move", "pct_12w_before", "vol_26w", "n_labelers", "next_12w_includes_january",
           "outcome_known", "target_pct_change", "moved_5pct"]
wl = wl[wl_cols].astype({"family_p": np.float32, "p_move": np.float32, "pct_12w_before": np.float32,
                         "vol_26w": np.float32, "target_pct_change": np.float32})
wl.sort_values(["origin_week", "family_rank", "p_move"], ascending=[True, True, False]) \
  .to_parquet(f"{OUT}/watchlist.parquet", index=False)
print(f"watch list: {len(wl):,} rows, {wl.origin_week.nunique()} origins "
      f"({(~wl.outcome_known).sum():,} rows without a known outcome)")

# ---------- prices and events ----------
gw = pd.read_parquet("data/tables/group_weekly", columns=["drug_group", "as_of_date", "median_price"])
pd.DataFrame({"gid": gw.drug_group.map(gid).astype(np.int32), "week": gw.as_of_date,
              "price": gw.median_price.astype(np.float32)}).sort_values(["gid", "week"]) \
  .to_parquet(f"{OUT}/prices.parquet", index=False)
pe = pd.read_parquet("data/tables/price_events", columns=["drug_group", "published_week", "old_price", "new_price",
                                                          "pct_change", "mix_only", "artefact_suspect"])
pe = pe.merge(ev[["drug_group", "published_week", "confidence"]], on=["drug_group", "published_week"], how="left")
pe["alert"] = np.select([pe.mix_only, pe.artefact_suspect], ["not scored: mix change", "not scored: artefact"],
                        default=pe.confidence.fillna("none"))
pd.DataFrame({"gid": pe.drug_group.map(gid).astype(np.int32), "week": pe.published_week,
              "old_price": pe.old_price.astype(np.float32), "new_price": pe.new_price.astype(np.float32),
              "pct_change": pe["pct_change"].astype(np.float32), "alert": pe.alert}) \
  .sort_values(["gid", "week"]).to_parquet(f"{OUT}/events.parquet", index=False)

# ---------- summary.json (numbers read from the metrics files) ----------
m8 = pd.read_csv("reports/metrics_8d.csv")
m10 = pd.read_csv("reports/metrics_forecast.csv")
m10b = pd.read_csv("reports/metrics_followups.csv")
m10c = pd.read_csv("reports/metrics_competition.csv")


def m8get(detector, metric, unit="event"):
    v = m8[(m8.unit == unit) & (m8.detector == detector) & (m8.metric == metric)].value
    return float(v.iloc[0])


test_alerts = al[al.period.str.startswith("2025")]
weekly = test_alerts.groupby("published_week").family.nunique()
summary = {
    "data": {"first_week": str(gw.as_of_date.min()), "last_week": str(gw.as_of_date.max()),
             "weeks": int(gw.as_of_date.nunique()), "drug_groups": int(len(groups)),
             "price_events": int(len(pe)), "ndc_rows_note": "CMS NADAC weekly files 2022-2026"},
    "store_brand_prefixes": prefixes,
    "anomaly_detectors": [
        {"detector": name, **{lab: m8get(det, met) for lab, met in [
            ("big_shift_recall_top100", "recall_big_shift@100"), ("relative_recall_top100", "recall_relative@100"),
            ("known_positive_share_top100", "known_pos_share@100"), ("hard_volatile_flagged_top1000", "hard_ff_volatile@1000"),
            ("january_list_increases_flagged_top1000", "ff_january_all@1000"), ("normal_moves_flagged_top1000", "ff_normal@1000")]}}
        for name, det in [("Rank by size of change (baseline)", "baseline_abs_pct"), ("Robust z-score (seasonal)", "zscore_B_seasonal"),
                          ("Hybrid (size + unusual for this drug)", "hybrid"), ("Isolation Forest, 3 features", "if_small"),
                          ("Isolation Forest, 13 features", "if_full"), ("Final: hybrid + IF second opinion", "final")]],
    "alerts_test_period": {"alert_events": int(len(test_alerts)), "families": int(test_alerts.family.nunique()),
                           "weeks": int((pd.to_datetime(pd.Series(gw.as_of_date.unique())) >= "2025-01-01").sum()),
                           "weeks_with_alerts": int(len(weekly)),
                           "max_families_per_week": int(weekly.max())},
    "forecast_regression": [
        {"model": name, "mae": float(r.mae), "skill": float(r.skill), "segment": r.segment}
        for name, mdl in [("Predict no change", "naive"), ("Drift", "drift"), ("Seasonal (brands, January)", "seasonal"),
                          ("LightGBM", "lgbm")]
        for _, r in m10[(m10.set == "backtest all folds") & (m10.rows_kind == "all rows") & (m10.model == mdl)].iterrows()],
    "forecast_regression_fix_skill": {r.segment: float(r.skill) for _, r in
                                      m10b[(m10b.part == "A") & (m10b.fold == "all") & (m10b.model == "shrunk (chosen alpha)")].iterrows()},
    "move_classifier": [
        {"model": {"vol rule": "Rank by recent volatility", "without": "Classifier (Step 10b)",
                   "with": "Classifier + competition features (used here)"}[r.model],
         "segment": r.segment, "roc_auc": float(r.roc_auc), "precision_top100": float(r.prec_at_100),
         "precision_top500": float(r.prec_at_500), "base_rate": float(r.rate)}
        for _, r in m10c[(m10c.fold == "all") & (m10c.target == "move >=5%")].iterrows()],
}
with open(f"{OUT}/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

sizes = {fn: os.path.getsize(f"{OUT}/{fn}") for fn in sorted(os.listdir(OUT))}
for fn, b in sizes.items():
    print(f"  {fn:20s} {b / 1e6:7.2f} MB")
print(f"total: {sum(sizes.values()) / 1e6:.2f} MB")
