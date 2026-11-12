"""Evaluate anomaly detectors against the step 8a evaluation set (data/eval/eval_events).

Reusable: add a detector to DETECTORS as a SQL query returning (drug_group, published_week, score), where a
higher score means more suspicious. Ranking pool: real events only (mix_only excluded).

Reports (see reports/anomaly_eval_plan.md):
  - test split (2025-26), for events AND clusters (same published week + same pct_change rounded to 0.1):
    recall@N per positive type, expected-event false flags@N, known-positive share@N, average precision
  - reverted spikes on ALL years, descriptive only (percentile rank)
  - named test cases (percentile rank among all years) and the top 20 test-split flags of chosen detectors
Metrics are saved to reports/metrics_<name>.csv.
"""
import csv
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

RUN_NAME = "zscore"
DETECTORS = {
    "baseline_abs_pct": "SELECT drug_group, published_week, ABS(pct_change) AS score FROM ev",
    "zscore_A_plain":   "SELECT drug_group, published_week, score_a AS score FROM zs",
    "zscore_B_seasonal": "SELECT drug_group, published_week, score_b AS score FROM zs",
}
TOP20_FOR = ["zscore_A_plain", "zscore_B_seasonal"]
TOP_N = [100, 500, 1000]

spark = (SparkSession.builder.master("local[*]").appName("evaluate")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.read.parquet("data/eval/eval_events").createOrReplaceTempView("ev")
spark.read.parquet("data/eval/scores_zscore").createOrReplaceTempView("zs")


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def q(sql, n=60):
    spark.sql(sql).show(n, truncate=False)


# every detector's score next to the labels; ties broken by a fixed hash so results are repeatable
union = " UNION ALL ".join(f"SELECT '{name}' AS detector, * FROM ({sql})" for name, sql in DETECTORS.items())
spark.sql(f"""
    SELECT s.detector, s.score, e.*,
           (e.is_pos_big_shift OR e.is_pos_big_spike OR e.is_pos_relative) AS is_pos_any,
           ROW_NUMBER() OVER (PARTITION BY s.detector
                              ORDER BY s.score DESC NULLS LAST, XXHASH64(e.drug_group, e.published_week)) AS rank_all,
           COUNT(*) OVER (PARTITION BY s.detector)                                                     AS n_all
    FROM ({union}) AS s
    JOIN ev AS e USING (drug_group, published_week)
    WHERE NOT e.mix_only
""").cache().createOrReplaceTempView("scored_base")
spark.sql("SELECT *, ROUND((rank_all - 1) * 100 / n_all, 2) AS top_pct_all FROM scored_base") \
    .createOrReplaceTempView("scored")

# units on the test split: single events, and clusters of events (same week, same rounded % change)
spark.sql("""
    WITH units AS (
        SELECT detector, 'event' AS unit, CONCAT(drug_group, '|', published_week) AS unit_id, 1 AS size, score,
               is_pos_big_shift AS pos_shift, is_pos_big_spike AS pos_spike, is_pos_relative AS pos_rel,
               is_neg_january_list_increase AS neg_jan, is_neg_normal_move AS neg_normal
        FROM scored WHERE split = 'test'
        UNION ALL
        SELECT detector, 'cluster', CONCAT(published_week, '|', ROUND(pct_change, 1)), COUNT(*), MAX(score),
               BOOL_OR(is_pos_big_shift), BOOL_OR(is_pos_big_spike), BOOL_OR(is_pos_relative),
               BOOL_OR(is_neg_january_list_increase) AND NOT BOOL_OR(is_pos_any),
               BOOL_OR(is_neg_normal_move) AND NOT BOOL_OR(is_pos_any)
        FROM scored WHERE split = 'test'
        GROUP BY detector, published_week, ROUND(pct_change, 1))
    SELECT *, ROW_NUMBER() OVER (PARTITION BY detector, unit ORDER BY score DESC NULLS LAST, XXHASH64(unit_id)) AS rnk
    FROM units
""").cache().createOrReplaceTempView("units")

header("test split: how many units of each label")
q("""SELECT unit, COUNT(*) AS units, SUM(CAST(pos_shift AS INT)) AS big_shift, SUM(CAST(pos_rel AS INT)) AS relative,
            SUM(CAST(pos_spike AS INT)) AS spike, SUM(CAST(neg_jan AS INT)) AS neg_january,
            SUM(CAST(neg_normal AS INT)) AS neg_normal, MAX(size) AS largest_cluster
     FROM units WHERE detector = 'baseline_abs_pct' GROUP BY unit ORDER BY unit DESC""")

n_values = ", ".join(str(n) for n in TOP_N)
at_n = spark.sql(f"""
    SELECT u.unit, u.detector, t.n AS top_n,
        ROUND(SUM(CASE WHEN pos_shift AND rnk <= t.n THEN 1 ELSE 0 END) / SUM(CAST(pos_shift AS INT)), 3) AS recall_big_shift,
        ROUND(SUM(CASE WHEN pos_rel   AND rnk <= t.n THEN 1 ELSE 0 END) / SUM(CAST(pos_rel   AS INT)), 3) AS recall_relative,
        SUM(CASE WHEN neg_jan    AND rnk <= t.n THEN 1 ELSE 0 END)                                       AS false_flags_january,
        SUM(CASE WHEN neg_normal AND rnk <= t.n THEN 1 ELSE 0 END)                                       AS false_flags_normal,
        ROUND(SUM(CASE WHEN (pos_shift OR pos_spike OR pos_rel) AND rnk <= t.n THEN 1 ELSE 0 END) / t.n, 3)
                                                                                                         AS known_pos_share
    FROM units AS u CROSS JOIN (SELECT EXPLODE(ARRAY({n_values})) AS n) AS t
    GROUP BY u.unit, u.detector, t.n
    ORDER BY u.unit DESC, t.n, u.detector""")
header("test split: metrics in the top N flags")
at_n.show(60, truncate=False)

# average precision: positives of one type vs the known negatives, in each detector's order
ap = spark.sql("""
    WITH typed AS (
        SELECT 'big_shift' AS pos_type, detector, unit, rnk, pos_shift AS is_pos FROM units
        WHERE pos_shift OR neg_jan OR neg_normal
        UNION ALL
        SELECT 'relative', detector, unit, rnk, pos_rel FROM units
        WHERE pos_rel OR neg_jan OR neg_normal),
    cum AS (
        SELECT *, SUM(CAST(is_pos AS INT)) OVER (PARTITION BY pos_type, detector, unit ORDER BY rnk) AS pos_so_far,
                  ROW_NUMBER()            OVER (PARTITION BY pos_type, detector, unit ORDER BY rnk) AS pool_rank
        FROM typed)
    SELECT unit, detector,
           ROUND(AVG(CASE WHEN pos_type = 'big_shift' AND is_pos THEN pos_so_far / pool_rank END), 3) AS ap_big_shift,
           ROUND(AVG(CASE WHEN pos_type = 'relative'  AND is_pos THEN pos_so_far / pool_rank END), 3) AS ap_relative
    FROM cum GROUP BY unit, detector ORDER BY unit DESC, detector""")
header("test split: average precision (1.0 = every positive ranked above every known negative)")
ap.show(truncate=False)

header("reverted spikes, ALL years (descriptive): percentile rank among all real events, 0 = top")
spikes = spark.sql("""
    SELECT detector, COUNT(*) AS spikes,
           PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY top_pct_all) AS median_top_pct,
           SUM(CASE WHEN top_pct_all <= 1 THEN 1 ELSE 0 END)        AS in_top_1pct,
           SUM(CASE WHEN top_pct_all <= 5 THEN 1 ELSE 0 END)        AS in_top_5pct
    FROM scored WHERE is_pos_big_spike GROUP BY detector ORDER BY detector""")
spikes.show(truncate=False)

header("named test cases: top_pct among all years (pass: 'flag' in top 1%, 'no_flag' outside top 10%)")
pivot = ",\n".join(
    f"MAX(CASE WHEN detector = '{d}' THEN top_pct_all END) AS `{d}`" for d in DETECTORS)
passes = ",\n".join(
    f"""MAX(CASE WHEN detector = '{d}' THEN CASE WHEN test_case_expect = 'flag' THEN top_pct_all <= 1
                                           ELSE top_pct_all > 10 END END) AS `pass_{d.split('_')[0]}_{d.split('_')[1]}`"""
    for d in DETECTORS)
q(f"""SELECT test_case, test_case_expect AS expect, published_week, ROUND(pct_change, 1) AS pct_change,
             {pivot}, {passes}
      FROM scored WHERE test_case_expect IS NOT NULL
      GROUP BY test_case, test_case_expect, published_week, pct_change ORDER BY test_case, published_week""")

for name in TOP20_FOR:
    variant = name.split("_")[1].lower()   # 'a' or 'b'
    others = [d for d in TOP20_FOR if d != name]
    header(f"top 20 flags, test split: {name}")
    q(f"""WITH top AS (
              SELECT detector, drug_group, published_week, ROW_NUMBER() OVER (PARTITION BY detector ORDER BY rank_all) AS r
              FROM scored WHERE split = 'test')
          SELECT t.r AS rank, s.drug_group, s.published_week, s.old_price, s.new_price, ROUND(s.pct_change, 1) AS pct,
                 z.z_{variant} AS z, z.n_hist_{variant} AS n_hist, z.fallback_{variant} AS fallback,
                 z.scale_floored_{variant} AS floored, s.category,
                 o.drug_group IS NOT NULL AS in_other_top20
          FROM top AS t
          JOIN scored AS s ON s.detector = t.detector AND s.drug_group = t.drug_group
                          AND s.published_week = t.published_week
          JOIN zs AS z ON z.drug_group = t.drug_group AND z.published_week = t.published_week
          LEFT JOIN (SELECT DISTINCT drug_group, published_week FROM top
                     WHERE detector IN ({", ".join(f"'{o}'" for o in others)}) AND r <= 20) AS o
                 ON o.drug_group = t.drug_group AND o.published_week = t.published_week
          WHERE t.detector = '{name}' AND t.r <= 20 ORDER BY t.r""")

# save every metric in long form, so later methods can be compared side by side
os.makedirs("reports", exist_ok=True)
with open(f"reports/metrics_{RUN_NAME}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split", "unit", "detector", "metric", "value"])
    for r in at_n.collect():
        for m in ["recall_big_shift", "recall_relative", "false_flags_january", "false_flags_normal", "known_pos_share"]:
            w.writerow(["test", r["unit"], r["detector"], f"{m}@{r['top_n']}", r[m]])
    for r in ap.collect():
        for m in ["ap_big_shift", "ap_relative"]:
            w.writerow(["test", r["unit"], r["detector"], m, r[m]])
    for r in spikes.collect():
        for m in ["median_top_pct", "in_top_1pct", "in_top_5pct"]:
            w.writerow(["all", "event", r["detector"], f"spike_{m}", r[m]])
print(f"\nsaved reports/metrics_{RUN_NAME}.csv")
spark.stop()
