"""Step 9: build the forecasting dataset, check it for leakage, describe the target. No model yet.

  1. runs sql/forecast_dataset.sql, keeps rows whose t+12 price exists -> data/tables/forecast_dataset/
  2. leakage checks:
       a. provenance: the latest week any feature read (feat_max_src_wk) is never after the origin
       b. truncation test: for a random sample of rows, recompute every feature in pandas from data CUT OFF at
          week t (later weeks deleted) and compare with the table. Equal values = the feature can't have used
          anything after t.
       c. purge: no training target window reaches the test period, also for each backtest fold
  3. target description and the "predict no change" baseline (MAE of y_hat = 0)
"""
import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

TEST_START = "2025-01-01"
FOLD_STARTS = {1: "2025-01-01", 2: "2025-06-18", 3: "2025-12-03", 4: "2026-04-22"}
SAMPLE_ROWS = 400
SEED = 42

spark = (SparkSession.builder.master("local[*]").appName("forecast_setup")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
for t in ["group_weekly", "price_events", "group_dim"]:
    spark.read.parquet(f"data/tables/{t}").createOrReplaceTempView(t)


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def q(sql, n=60):
    spark.sql(sql).show(n, truncate=False)


# ---------- 1. build ----------
spark.sql(Path("sql/forecast_dataset.sql").read_text(encoding="utf-8")).cache().createOrReplaceTempView("cand")
header("candidate rows (group priced at t, >= 26 priced weeks) by target status")
q("""SELECT target_status, COUNT(*) AS rows, COUNT(DISTINCT origin_week) AS origins,
            MIN(origin_week) AS first_origin, MAX(origin_week) AS last_origin
     FROM cand GROUP BY 1 ORDER BY 1""")
q("""SELECT split, COUNT(*) AS rows_missing_target_at_t_plus_12
     FROM cand WHERE target_status = 'not_priced_at_t_plus_12' GROUP BY 1 ORDER BY 1""")

spark.sql("SELECT * FROM cand WHERE target_status = 'ok'").drop("target_status") \
     .write.mode("overwrite").parquet("data/tables/forecast_dataset")
spark.read.parquet("data/tables/forecast_dataset").createOrReplaceTempView("fd")
# Step 11: the newest origins have no t+12 price yet. Saved separately (never trained or evaluated on) so the
# dashboard can show a "current" watch list. Same features, target columns are NULL.
spark.sql("SELECT * FROM cand WHERE target_status = 'beyond_data_end'").drop("target_status") \
     .write.mode("overwrite").parquet("data/tables/forecast_latest")

header("forecast_dataset: rows per split / fold")
q("""SELECT split, COALESCE(CAST(fold AS STRING), '-') AS fold, COUNT(*) AS rows,
            COUNT(DISTINCT origin_week) AS origins, COUNT(DISTINCT drug_group) AS groups,
            MIN(origin_week) AS first_origin, MAX(origin_week) AS last_origin, MAX(target_week) AS last_target_week,
            SUM(CAST(tgt_has_artefact AS INT)) AS tgt_has_artefact, SUM(CAST(tgt_has_mix_only AS INT)) AS tgt_has_mix_only
     FROM fd GROUP BY 1, 2 ORDER BY MIN(origin_week)""")

header("market_move_4w (median 4-week log change of generics) across origins")
q("""SELECT COUNT(DISTINCT origin_week) AS origins,
            SUM(CASE WHEN m = 0 THEN 1 ELSE 0 END) AS origins_where_median_is_exactly_0,
            ROUND(MIN(m), 5) AS min, ROUND(MAX(m), 5) AS max,
            ROUND(MIN(s), 3) AS min_share_moved, ROUND(PERCENTILE(s, 0.5), 3) AS median_share_moved,
            ROUND(MAX(s), 3) AS max_share_moved
     FROM (SELECT DISTINCT origin_week, market_move_4w AS m, market_share_moved_4w AS s FROM fd)""")

header("feature summary (all rows)")
FEATURES = ["chg_4w", "chg_12w", "chg_26w", "chg_52w", "vol_26w", "n_changes_26w", "weeks_since_change",
            "last_change_size", "log_price", "n_ndcs", "n_distinct_prices", "month", "market_move_4w"]
q(" UNION ALL ".join(
    f"""SELECT '{c}' AS feature, COUNT({c}) AS non_null, SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS nulls,
               ROUND(PERCENTILE({c}, 0.01), 4) AS p01, ROUND(PERCENTILE({c}, 0.5), 4) AS p50,
               ROUND(PERCENTILE({c}, 0.99), 4) AS p99 FROM fd""" for c in FEATURES))
q("""SELECT ROUND(AVG(CAST(is_brand AS INT)), 3) AS brand_share, ROUND(AVG(CAST(no_change_seen AS INT)), 3) AS no_change_seen,
            ROUND(AVG(CAST(chg_52w_missing AS INT)), 3) AS chg_52w_missing,
            ROUND(AVG(CAST(next_12w_includes_january AS INT)), 3) AS next_12w_includes_january
     FROM fd""")
q("SELECT classification, pricing_unit, COUNT(*) AS rows FROM fd GROUP BY 1, 2 ORDER BY 3 DESC")

# ---------- 2. leakage checks ----------
header("leakage check a: provenance (latest source week a feature read vs origin week)")
q("""SELECT COUNT(*) AS rows, MAX(feat_max_src_wk - origin_wk) AS max_src_minus_origin,
            SUM(CASE WHEN feat_max_src_wk > origin_wk THEN 1 ELSE 0 END) AS violations
     FROM fd""")

header("leakage check c: purge (training target windows must end before the test period / fold)")
q(f"""SELECT 'final split' AS check, '{TEST_START}' AS test_start,
             SUM(CASE WHEN split = 'train' THEN 1 ELSE 0 END) AS train_rows,
             MAX(CASE WHEN split = 'train' THEN target_week END) AS last_train_target_week,
             SUM(CASE WHEN split = 'train' AND target_week >= DATE'{TEST_START}' THEN 1 ELSE 0 END) AS overlapping_rows,
             SUM(CASE WHEN split = 'test' THEN 1 ELSE 0 END) AS test_rows
      FROM fd""")
q(" UNION ALL ".join(f"""
    SELECT 'fold {k}' AS check, '{s}' AS test_start,
           SUM(CASE WHEN target_week < DATE'{s}' THEN 1 ELSE 0 END) AS train_rows,
           MAX(CASE WHEN target_week < DATE'{s}' THEN target_week END) AS last_train_target_week,
           MAX(CASE WHEN target_week < DATE'{s}' THEN origin_week END) AS last_train_origin,
           SUM(CASE WHEN target_week < DATE'{s}' AND origin_week >= DATE'{s}' THEN 1 ELSE 0 END) AS overlapping_rows,
           SUM(CASE WHEN fold = {k} THEN 1 ELSE 0 END) AS test_rows
    FROM fd""" for k, s in FOLD_STARTS.items()))

# truncation test (pandas): recompute features from data with every week after t deleted
sample = spark.table("fd").sample(fraction=0.002, seed=SEED).limit(SAMPLE_ROWS).toPandas()
sample_origins = sorted(sample.origin_week.unique())
gw = spark.sql("""
    SELECT g.drug_group, g.as_of_date, g.median_price, g.n_ndcs_priced, g.n_distinct_prices, g.price_changed,
           COALESCE(e.mix_only, FALSE) AS mix_only, COALESCE(e.artefact_suspect, FALSE) AS artefact_suspect,
           d.classification
    FROM group_weekly g JOIN group_dim d USING (drug_group)
    LEFT JOIN price_events e ON e.drug_group = g.drug_group AND e.published_week = g.as_of_date""").toPandas()
spark.stop()

weeks = sorted(gw.as_of_date.unique())
wk_of = {w: i for i, w in enumerate(weeks)}
gw["wk"] = gw.as_of_date.map(wk_of)
gw = gw.sort_values(["drug_group", "wk"])
by_group = dict(tuple(gw.groupby("drug_group")))


def features_from_truncated(g, t_wk):
    """All features of group g at origin t_wk, using ONLY rows with wk <= t_wk."""
    h = g[g.wk <= t_wk].copy()                       # <-- everything after t is deleted here
    h["lp"] = np.log(h.median_price)
    h["dl"] = h.lp.diff()
    h["real"] = (h.price_changed == True) & ~h.mix_only & ~h.artefact_suspect   # NULL (first week) -> False
    now = h.iloc[-1]
    out = {}
    for k in [4, 12, 26, 52]:
        past = h[h.wk <= t_wk - k]
        out[f"chg_{k}w"] = now.lp - past.lp.iloc[-1] if len(past) else np.nan
    win = h[h.wk >= t_wk - 25]
    out["vol_26w"] = win.dl.std(ddof=1)
    out["n_changes_26w"] = int(win.real.sum())
    ch = h[h.real]
    out["weeks_since_change"] = t_wk - (ch.wk.iloc[-1] if len(ch) else g.wk.min())
    out["last_change_size"] = ch.dl.iloc[-1] if len(ch) else 0.0
    out["log_price"] = now.lp
    out["n_ndcs"] = now.n_ndcs_priced
    out["n_distinct_prices"] = now.n_distinct_prices
    return out


# NB: weeks_since_change for a never-changed group counts from the group's first week (g.wk.min()), which is <= t.
mismatch, compared = {}, 0
for _, r in sample.iterrows():
    rec = features_from_truncated(by_group[r.drug_group], r.origin_wk)
    for c, v in rec.items():
        compared += 1
        a, b = r[c], v
        same = (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and abs(float(a) - float(b)) < 1e-9)
        if not same:
            mismatch.setdefault(c, []).append((r.drug_group, r.origin_week, a, b))

# market move for the sampled origins, from generics truncated at t
mk_mismatch = 0
gen = gw[gw.classification == "G"]
for t in sample_origins:
    t_wk = wk_of[t]
    h = gen[gen.wk <= t_wk]
    now = h[h.wk == t_wk].set_index("drug_group").median_price
    past = h[h.wk <= t_wk - 4].groupby("drug_group").median_price.last()
    both = now.index.intersection(past.index)
    m = np.median(np.log(now[both] / past[both]))
    table_m = sample.loc[sample.origin_week == t, "market_move_4w"].iloc[0]
    mk_mismatch += abs(m - table_m) > 1e-9   # Spark PERCENTILE interpolates like np.median

header("leakage check b: truncation test (features recomputed from data cut off at week t)")
print(f"sampled rows: {len(sample)}, origins: {len(sample_origins)}, feature values compared: {compared:,}")
print(f"mismatches: {sum(len(v) for v in mismatch.values())}"
      + "".join(f"\n  {c}: {len(v)} e.g. {v[:2]}" for c, v in mismatch.items()))
print(f"market_move_4w origins compared: {len(sample_origins)}, mismatches: {mk_mismatch}")

# ---------- 3. target description ----------
fd = pd.read_parquet("data/tables/forecast_dataset")
fd = fd[fd.split != "purged"]
y = "target_log_change"


def pct_table(df, by):
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    t = df.groupby(by)[y].quantile(qs).unstack().round(4)
    t.columns = [f"p{int(q * 100):02d}" for q in qs]
    t.insert(0, "rows", df.groupby(by).size())
    t["exactly_0"] = df.groupby(by)[y].apply(lambda s: (s == 0).mean()).round(3)
    return t


pd.set_option("display.width", 200)
header("target: ln(price t+12 / price t), percentiles by split and brand")
print(pct_table(fd, ["split", "is_brand"]).to_string())

header("target direction (up / down / flat within +-2%) by split and brand")
print((fd.groupby(["split", "is_brand"]).target_direction.value_counts(normalize=True).unstack().round(3)).to_string())

header("brands: does the next 12 weeks include a January?")
b = fd[fd.is_brand]
t = b.groupby(["split", "next_12w_includes_january"]).agg(
    rows=(y, "size"), mean_target=(y, "mean"), share_up=("target_direction", lambda s: (s == "up").mean()),
    share_exactly_0=(y, lambda s: (s == 0).mean())).round(4)
t["share_of_brand_rows"] = (t.rows / t.groupby(level=0).rows.transform("sum")).round(3)
print(t.to_string())

header("baseline 'predict no change' (y_hat = 0): MAE = mean |target|")
rows = []
for name, df in [("train", fd[fd.split == "train"]), ("test", fd[fd.split == "test"])] + \
                [(f"fold {k}", fd[fd.fold == k]) for k in FOLD_STARTS]:
    clean = df[~df.tgt_has_artefact]
    rows.append({"set": name, "rows": len(df), "mae_all": df[y].abs().mean(),
                 "mae_generic": df[~df.is_brand][y].abs().mean(), "mae_brand": df[df.is_brand][y].abs().mean(),
                 "rows_no_artefact": len(clean), "mae_no_artefact": clean[y].abs().mean(),
                 "median_abs": df[y].abs().median()})
print(pd.DataFrame(rows).round(5).to_string(index=False))
