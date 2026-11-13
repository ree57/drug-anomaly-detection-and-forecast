"""Step 10c: competition (labeler) features in the Step 10b move classifier. One run.

1. signal check on the TRAIN split only: share of rows moving >= 5% (and up / down) by n_labelers and after
   labeler exits / entries
2. the Step 10b classifier (same features, folds, purging, expanding window, early stopping on validation, refit)
   WITHOUT vs WITH the competition features; the version kept is the one with the better mean VALIDATION ROC AUC
3. test comparison: all / generic / brand, and separately for increases (>= +5%) and decreases (<= -5%)
4. watch list for one origin grouped into product families (same rule as alerts.py: same name stem, or the same
   big recent % change), so one product can't fill several slots
Output: reports/metrics_competition.csv (tables printed for reports/forecast_competition.md).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
FOLD_STARTS = {1: "2025-01-01", 2: "2025-06-18", 3: "2025-12-03", 4: "2026-04-22"}
VAL_ORIGINS = 3
MOVE_PCT = 5
EXAMPLE_ORIGIN = "2026-04-22"
FAMILY_PCT_MIN = 30                  # as in alerts/evaluate: identical % change links products only for big moves
BASE = ["chg_4w", "chg_12w", "chg_26w", "chg_52w", "chg_52w_missing", "vol_26w", "n_changes_26w",
        "weeks_since_change", "last_change_size", "no_change_seen", "is_brand", "classification", "pricing_unit",
        "log_price", "n_ndcs", "n_distinct_prices", "next_12w_includes_january", "market_move_4w"]   # Step 10b
COMP = ["n_labelers", "d_labelers_12w", "d_labelers_26w", "labeler_exit_26w", "labeler_entry_26w", "single_source"]
VERSIONS = {"without": BASE, "with competition": BASE + COMP}
CATEGORICAL = ["classification", "pricing_unit"]
COMMON = dict(learning_rate=0.05, num_leaves=31, min_child_samples=200, colsample_bytree=0.8, random_state=SEED,
              deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


df = pd.read_parquet("data/tables/forecast_dataset")
comp = pd.read_parquet("data/tables/forecast_competition").drop(columns=["origin_wk", "comp_max_src_wk"])
df = df.merge(comp, on=["drug_group", "origin_week"], how="left", validate="one_to_one")
df = df[df.split != "purged"].copy()
for c in ["origin_week", "target_week"]:
    df[c] = pd.to_datetime(df[c])
for c in CATEGORICAL:
    df[c] = df[c].astype("category")
for c in ["chg_52w_missing", "no_change_seen", "is_brand", "next_12w_includes_january"]:
    df[c] = df[c].astype(int)
df["moves_5pct"] = (df.target_pct_change.abs() >= MOVE_PCT).astype(int)
df["up_5pct"] = (df.target_pct_change >= MOVE_PCT).astype(int)
df["down_5pct"] = (df.target_pct_change <= -MOVE_PCT).astype(int)
SEGMENTS = [("all", lambda d: d.is_brand >= 0), ("generic", lambda d: d.is_brand == 0),
            ("brand", lambda d: d.is_brand == 1)]

# ---------- 1. signal check (train only) ----------
tr = df[df.split == "train"].copy()
tr["labelers"] = np.where(tr.n_labelers >= 4, "4+", tr.n_labelers.astype(str))
rates = dict(rows=("moves_5pct", "size"), moves_5pct=("moves_5pct", "mean"), up_5pct=("up_5pct", "mean"),
             down_5pct=("down_5pct", "mean"))
header("signal check, TRAIN split: share of rows moving >= 5% in the next 12 weeks, by number of labelers at t")
print(tr.groupby(["is_brand", "labelers"]).agg(**rates).round(3).to_string())
header("signal check, TRAIN split: after labeler exits / entries in the last 26 weeks")
tr["exit"] = np.where(tr.labeler_exit_26w.isna(), "unknown", np.where(tr.labeler_exit_26w > 0, ">=1 exit", "no exit"))
tr["entry"] = np.where(tr.labeler_entry_26w.isna(), "unknown", np.where(tr.labeler_entry_26w > 0, ">=1 entry", "no entry"))
print(tr.groupby(["is_brand", "exit"]).agg(**rates).round(3).to_string())
print(tr.groupby(["is_brand", "entry"]).agg(**rates).round(3).to_string())
print("generic, single source vs not, by exit:")
g = tr[tr.is_brand == 0]
print(g.groupby([g.single_source == 1, "exit"]).agg(**rates).round(3).to_string())


# ---------- 2. with vs without ----------
def split_fold(start):
    pool = df[(df.target_week < pd.Timestamp(start)) & ~df.tgt_has_artefact]
    val_origins = sorted(pool.origin_week.unique())[-VAL_ORIGINS:]
    return pool[pool.target_week < val_origins[0]], pool[pool.origin_week.isin(val_origins)], pool


val_rows, models = [], {}
for version, feats in VERSIONS.items():
    for k, s in FOLD_STARTS.items():
        fit, val, refit = split_fold(s)
        m = lgb.LGBMClassifier(objective="binary", n_estimators=3000, **COMMON)
        m.fit(fit[feats], fit.moves_5pct, eval_X=(val[feats],), eval_y=(val.moves_5pct,),
              eval_metric="binary_logloss", categorical_feature=CATEGORICAL,
              callbacks=[lgb.early_stopping(100, verbose=False)])
        pv = m.predict_proba(val[feats], num_iteration=m.best_iteration_)[:, 1]
        vg = (val.is_brand == 0).to_numpy()
        val_rows.append({"version": version, "fold": k, "best_iter": m.best_iteration_,
                         "val_logloss": m.best_score_["valid_0"]["binary_logloss"],
                         "val_auc": roc_auc_score(val.moves_5pct, pv),
                         "val_auc_generic": roc_auc_score(val.moves_5pct[vg], pv[vg])})
        final = lgb.LGBMClassifier(objective="binary", n_estimators=m.best_iteration_, **COMMON)
        final.fit(refit[feats], refit.moves_5pct, categorical_feature=CATEGORICAL)
        df.loc[df.fold == k, f"p_{version}"] = final.predict_proba(df.loc[df.fold == k, feats])[:, 1]
        models[(version, k)] = final
vr = pd.DataFrame(val_rows)
header("VALIDATION results (the only basis for the choice)")
print(vr.round(4).to_string(index=False))
summary = vr.groupby("version")[["val_auc", "val_auc_generic", "val_logloss"]].mean()
print(summary.round(4).to_string())
kept = summary.val_auc.idxmax()
print(f"-> keep: {kept} (higher mean validation ROC AUC)")
df["p_vol_rule"] = df.vol_26w.fillna(0)


# ---------- 3. test comparison ----------
def prec_at(sub, col, n, label):
    return sub.sort_values(col, ascending=False, kind="stable").head(n)[label].mean()


rows = []
for k in list(FOLD_STARTS) + ["all"]:
    sel = df.fold.notna() if k == "all" else df.fold == k
    for seg, segf in SEGMENTS:
        sub = df[sel & segf(df)]
        for target, label, pool in [("move >=5%", "moves_5pct", sub),
                                    # direction: that direction vs rows that did not move >= 5% (other direction left out)
                                    ("increase >=+5%", "up_5pct", sub[sub.down_5pct == 0]),
                                    ("decrease <=-5%", "down_5pct", sub[sub.up_5pct == 0])]:
            for name, col in [("vol rule", "p_vol_rule"), ("without", "p_without"), ("with", "p_with competition")]:
                rows.append({"fold": k, "segment": seg, "target": target, "model": name, "rows": len(pool),
                             "rate": pool[label].mean(), "roc_auc": roc_auc_score(pool[label], pool[col]),
                             "pr_auc": average_precision_score(pool[label], pool[col]),
                             # precision@N on the FULL fold-segment list: share of the top N that were this outcome
                             "prec_at_100": prec_at(sub, col, 100, label), "prec_at_500": prec_at(sub, col, 500, label)})
res = pd.DataFrame(rows)
header("TEST: 4 folds pooled (rate = share of positives; direction rows exclude the opposite direction)")
pooled = res[res.fold == "all"].pivot_table(index=["target", "segment"], columns="model",
                                            values=["rate", "roc_auc", "pr_auc", "prec_at_100", "prec_at_500"], sort=False)
print(pooled.round(3).to_string())
header("TEST: generic, per fold (move >= 5%)")
print(res[(res.segment == "generic") & (res.target == "move >=5%") & (res.fold != "all")]
      .pivot_table(index="fold", columns="model", values=["roc_auc", "prec_at_100", "prec_at_500"]).round(3).to_string())
header("TEST: per fold, all segments, move >= 5%: ROC AUC without -> with")
t = res[(res.target == "move >=5%") & (res.model != "vol rule")].pivot_table(index=["fold", "segment"], columns="model",
                                                                               values="roc_auc", sort=False)
t["diff"] = t["with"] - t["without"]
print(t.round(4).to_string())

header("competition features in the fold-4 model WITH them (gain share, rank among all features)")
m4 = models[("with competition", 4)]
gain = pd.Series(m4.booster_.feature_importance("gain"), index=VERSIONS["with competition"]).sort_values(ascending=False)
share = (gain / gain.sum()).round(4)
print(pd.DataFrame({"gain_share": share, "rank": range(1, len(share) + 1)}).loc[COMP].to_string())


# ---------- 4. family-grouped watch list ----------
def families(rows):
    """Union-find: same name stem, or (|pct_12w| >= FAMILY_PCT_MIN) the same 12-week % change to 2 decimals."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    pct = (np.exp(rows.chg_12w.to_numpy()) - 1) * 100
    stems = rows.drug_group.str.extract(r"^([^ |-]+)")[0].to_numpy()
    keys = {}
    for i in range(len(rows)):
        ks = [("n", stems[i])] + ([("p", round(pct[i], 2))] if abs(pct[i]) >= FAMILY_PCT_MIN else [])
        for key in ks:
            if key in keys:
                parent[find(i)] = find(keys[key])
            else:
                keys[key] = i
    return [find(i) for i in range(len(rows))]


col = f"p_{kept}"
ex = df[df.origin_week == pd.Timestamp(EXAMPLE_ORIGIN)].copy().reset_index(drop=True)
ex["family"] = families(ex)
fam = (ex.sort_values(col, ascending=False, kind="stable").groupby("family", sort=False)
         .agg(p_move=(col, "max"), members=("drug_group", "size"),
              names=("drug_group", lambda s: " / ".join(x.split("|")[0] for x in s.iloc[:3]) + (" …" if len(s) > 3 else "")),
              n_labelers=("n_labelers", "first"), exits_26w=("labeler_exit_26w", "first"),
              moved=("moves_5pct", "sum"), change_min=("target_pct_change", "min"), change_max=("target_pct_change", "max"))
         .sort_values("p_move", ascending=False, kind="stable"))
header(f"watch list at origin {EXAMPLE_ORIGIN}, grouped by product family ({kept} model, fold 4)")
print(f"drug groups: {len(ex):,} in {ex.family.nunique():,} families; share of groups moving >= 5%: {ex.moves_5pct.mean():.3f}")
top_rows = ex.sort_values(col, ascending=False, kind="stable").head(10)
print(f"ungrouped top 10: {top_rows.family.nunique()} distinct families, {top_rows.moves_5pct.sum()} moved")
top = fam.head(10)
print(f"grouped top 10 families: {int((top.moved > 0).sum())} with a member that moved; "
      f"members moved {int(top.moved.sum())}/{int(top.members.sum())}")
print(top.round(3).to_string(index=False))

res.to_csv("reports/metrics_competition.csv", index=False, float_format="%.6f")
vr.to_csv("reports/metrics_competition_validation.csv", index=False, float_format="%.6f")
print("\nsaved reports/metrics_competition.csv and reports/metrics_competition_validation.csv")
