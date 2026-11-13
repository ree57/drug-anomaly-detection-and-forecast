"""Step 8c: Method 2, Isolation Forest, in two versions (IF-small, IF-full).

One row per real event (mix_only and artefact_suspect excluded), features known at publication time only
(no outcome_ columns). Models are FITTED on events published 2022-2024 and then score every event:
2025-26 scores are out-of-sample; 2022-24 scores are in-sample (descriptive use only).

Spark builds the features; the table (~250k rows x 13 numbers, ~30 MB) is small enough for pandas/sklearn.
Score = -score_samples(): higher = shorter average path = more anomalous. Only the ranking is used,
so `contamination` (which only sets a yes/no threshold) is irrelevant and left at its default.

Output: data/eval/scores_iforest/ (features + both scores, also used by explain_iforest.py).
"""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.ensemble import IsolationForest

TRAIN_END = "2025-01-01"
HISTORY, MIN_HISTORY = 12, 6
SETTINGS = dict(n_estimators=200, max_samples=256, random_state=42, n_jobs=-1)
FEATURES_SMALL = ["abs_log_change", "z_b_signed_log", "typical_past_move"]
FEATURES_FULL = ["log_change", "abs_log_change", "z_b_signed_log", "typical_past_move", "weeks_since_last_event",
                 "is_brand", "is_january", "after_gap", "class_changed", "log_old_price", "n_ndcs_priced",
                 "n_distinct_prices", "has_code_3"]

spark = (SparkSession.builder.master("local[*]").appName("detect_iforest")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
for name in ["price_events", "group_weekly"]:
    spark.read.parquet(f"data/tables/{name}").createOrReplaceTempView(name)
spark.read.parquet("data/eval/scores_zscore").createOrReplaceTempView("zs")


def median_of(arr):
    s = f"ARRAY_SORT({arr})"
    lo = f"ELEMENT_AT({s}, CAST(FLOOR((SIZE({arr}) + 1) / 2) AS INT))"
    hi = f"ELEMENT_AT({s}, CAST(CEIL((SIZE({arr}) + 1) / 2) AS INT))"
    return f"CASE WHEN SIZE({arr}) > 0 THEN ({lo} + {hi}) / 2 END"


features = spark.sql(f"""
    WITH real AS (
        SELECT e.*, LN(e.new_price / e.old_price) AS x, TRUNC(e.published_week, 'MM') AS month
        FROM price_events AS e WHERE NOT e.mix_only AND NOT e.artefact_suspect),
    hist AS (   -- the group's previous 12 real moves, never the event itself
        SELECT *, COLLECT_LIST(ABS(x)) OVER (PARTITION BY drug_group ORDER BY published_week
                                             ROWS BETWEEN {HISTORY} PRECEDING AND 1 PRECEDING) AS past_abs
        FROM real),
    class_month AS (   -- fallback: typical move of the class, from events published before this month
        SELECT m.month, r.is_brand, MEDIAN(ABS(r.x)) AS class_typical
        FROM (SELECT DISTINCT month FROM real) AS m JOIN real AS r ON r.published_week < m.month
        GROUP BY m.month, r.is_brand),
    first_seen AS (SELECT drug_group, MIN(as_of_date) AS first_week FROM group_weekly GROUP BY drug_group)
    SELECT h.drug_group, h.published_week,
        CASE WHEN h.published_week < DATE'{TRAIN_END}' THEN 'train' ELSE 'test' END       AS split,
        h.x                                                                            AS log_change,
        ABS(h.x)                                                                       AS abs_log_change,
        SIGN(z.z_b) * LN(1 + ABS(z.z_b))                                               AS z_b_signed_log,
        IF(SIZE(h.past_abs) >= {MIN_HISTORY}, {median_of('h.past_abs')}, c.class_typical) AS typical_past_move,
        SIZE(h.past_abs) < {MIN_HISTORY}                                               AS typical_from_class,
        CAST(COALESCE(h.weeks_since_last_event,
                      CAST(DATEDIFF(h.published_week, f.first_week) / 7 AS INT)) AS DOUBLE) AS weeks_since_last_event,
        CAST(h.is_brand AS DOUBLE)      AS is_brand,
        CAST(h.is_january AS DOUBLE)    AS is_january,
        CAST(h.after_gap AS DOUBLE)     AS after_gap,
        CAST(h.class_changed AS DOUBLE) AS class_changed,
        LN(h.old_price)                 AS log_old_price,
        CAST(h.n_ndcs_priced AS DOUBLE) AS n_ndcs_priced,
        CAST(h.n_distinct_prices AS DOUBLE) AS n_distinct_prices,
        CAST(COALESCE(ARRAY_CONTAINS(h.explanation_codes, 3), FALSE) AS DOUBLE) AS has_code_3
    FROM hist AS h
    LEFT JOIN zs AS z USING (drug_group, published_week)
    LEFT JOIN class_month AS c ON c.month = h.month AND c.is_brand = h.is_brand
    LEFT JOIN first_seen AS f USING (drug_group)
""").toPandas()
print(f"feature rows: {len(features):,}  (train {sum(features.split == 'train'):,}, test {sum(features.split == 'test'):,})")

# missing values: leave incomplete rows out of training; when scoring, fill them with the train median (flagged)
train_mask = features.split == "train"
missing = features[FEATURES_FULL].isna().any(axis=1)
print(f"rows with a missing feature: train {int((missing & train_mask).sum()):,}, test {int((missing & ~train_mask).sum()):,}")
print("missing per feature:", {c: int(n) for c, n in features[FEATURES_FULL].isna().sum().items() if n})
fit_rows = features[train_mask & ~missing]
features["filled_missing"] = missing
filled = features[FEATURES_FULL].fillna(fit_rows[FEATURES_FULL].median())

for name, cols in [("if_small", FEATURES_SMALL), ("if_full", FEATURES_FULL)]:
    model = IsolationForest(**SETTINGS).fit(fit_rows[cols])
    features[f"score_{name}"] = -model.score_samples(filled[cols])
    print(f"{name}: fitted on {len(fit_rows):,} train events, {len(cols)} features")

print("\nscore summary (higher = more anomalous)")
print(features.groupby("split")[["score_if_small", "score_if_full"]].describe().T.round(3).to_string())

spark.createDataFrame(features).write.mode("overwrite").parquet("data/eval/scores_iforest")
print("\nsaved data/eval/scores_iforest")
spark.stop()
