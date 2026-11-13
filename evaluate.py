"""Evaluate anomaly detectors against the evaluation set (data/eval/eval_events).

Reusable: add a detector to DETECTORS as a SQL query returning (drug_group, published_week, score), where a
higher score means more suspicious. Ranking pool: events whose category is not excluded_* (mix_only, artefacts).

Reports (definitions in reports/anomaly_eval_plan.md):
  - test split (2025-26), for single events AND families (events of the same week linked by the same name
    stem or the same % change; see build_families):
    recall@N per positive type, false flags@N (expected + HARD negatives), known-positive share@N,
    average precision against easy negatives and against hard negatives
  - reverted spikes on ALL years, descriptive only (percentile rank)
  - named test cases (percentile rank among all years) and the top 20 test-split families of one detector
Metrics are saved to reports/metrics_<RUN_NAME>.csv.
"""
import csv
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

RUN_NAME = "8c"
DETECTORS = {
    "baseline_abs_pct":  "SELECT drug_group, published_week, ABS(pct_change) AS score FROM ev",
    "zscore_B_seasonal": "SELECT drug_group, published_week, score_b AS score FROM zs",
    "hybrid":            "SELECT drug_group, published_week, score_hybrid AS score FROM hy",
    "if_small":          "SELECT drug_group, published_week, score_if_small AS score FROM iforest",
    "if_full":           "SELECT drug_group, published_week, score_if_full AS score FROM iforest",
}
TOP_FAMILIES_FOR = ["if_full", "if_small"]
TOP_N = [100, 500, 1000]
SHORT = {"baseline_abs_pct": "base", "zscore_A_plain": "zA", "zscore_B_seasonal": "zB", "hybrid": "hyb",
         "if_small": "ifS", "if_full": "ifF"}   # short column names in the printed tables
FAMILY_PCT_MIN = 30   # only moves this big (%) can join a family through an identical % change ...
FAMILY_PCT_DECIMALS = 2   # ... rounded to this many decimals (10% / 0.1 chained 2,000+ unrelated generics)

spark = (SparkSession.builder.master("local[*]").appName("evaluate")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.read.parquet("data/eval/eval_events").createOrReplaceTempView("ev")
spark.read.parquet("data/eval/scores_zscore").createOrReplaceTempView("zs")
spark.read.parquet("data/eval/scores_hybrid").createOrReplaceTempView("hy")
spark.read.parquet("data/eval/scores_iforest").createOrReplaceTempView("iforest")


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def q(sql, n=60):
    spark.sql(sql).show(n, truncate=False)


def build_families():
    """Families: two events of the same week are linked if they share the first word of the description
    (split on space and '-') or, for moves >= FAMILY_PCT_MIN %, the same % change rounded to 0.1.
    Linked chains form one family: repeat "take the smallest label among my links" until nothing changes."""
    fam = spark.sql(f"""
        SELECT drug_group, published_week, CONCAT(drug_group, '|', published_week) AS label,
               CONCAT(published_week, '|', REGEXP_EXTRACT(drug_group, '^([^ |-]+)', 1)) AS k_name,
               CASE WHEN ABS(pct_change) >= {FAMILY_PCT_MIN}
                    THEN CONCAT(published_week, '|', ROUND(pct_change, {FAMILY_PCT_DECIMALS})) END AS k_pct
        FROM ev WHERE category NOT LIKE 'excluded%'""").localCheckpoint()
    for i in range(1, 21):
        fam.createOrReplaceTempView("fam")
        nxt = spark.sql("""
            WITH a AS (SELECT *, MIN(label) OVER (PARTITION BY k_name) AS l1 FROM fam)
            SELECT drug_group, published_week, k_name, k_pct, label AS old_label,
                   CASE WHEN k_pct IS NULL THEN l1 ELSE MIN(l1) OVER (PARTITION BY k_pct) END AS label
            FROM a""").localCheckpoint()
        changed = nxt.filter("label <> old_label").count()
        fam = nxt.drop("old_label")
        print(f"family pass {i}: {changed:,} labels changed")
        if changed == 0:
            break
    fam.select("drug_group", "published_week", fam.label.alias("family_id")) \
       .write.mode("overwrite").parquet("data/eval/families")   # reused by explain_iforest.py
    fam.createOrReplaceTempView("families")


build_families()

# every detector's score next to the labels and family; ties broken by a fixed hash so results repeat
union = " UNION ALL ".join(f"SELECT '{name}' AS detector, * FROM ({sql})" for name, sql in DETECTORS.items())
spark.sql(f"""
    SELECT s.detector, s.score, e.*, f.label AS family_id,
           (e.is_pos_big_shift OR e.is_pos_big_spike OR e.is_pos_relative) AS is_pos_any,
           ROW_NUMBER() OVER (PARTITION BY s.detector
                              ORDER BY s.score DESC NULLS LAST, XXHASH64(e.drug_group, e.published_week)) AS rank_all,
           COUNT(*) OVER (PARTITION BY s.detector)                                                     AS n_all
    FROM ({union}) AS s
    JOIN ev AS e USING (drug_group, published_week)
    JOIN families AS f USING (drug_group, published_week)
    WHERE e.category NOT LIKE 'excluded%'
""").cache().createOrReplaceTempView("scored_base")
spark.sql("SELECT *, ROUND((rank_all - 1) * 100 / n_all, 2) AS top_pct_all FROM scored_base") \
    .createOrReplaceTempView("scored")

# units on the test split: single events, and families
spark.sql("""
    WITH units AS (
        SELECT detector, 'event' AS unit, CONCAT(drug_group, '|', published_week) AS unit_id, 1 AS size, score,
               is_pos_big_shift AS pos_shift, is_pos_big_spike AS pos_spike, is_pos_relative AS pos_rel,
               is_neg_january_list_increase AS neg_jan, is_neg_normal_move AS neg_normal,
               is_neg_hard_january AS hard_jan, is_neg_hard_volatile AS hard_vol
        FROM scored WHERE split = 'test'
        UNION ALL
        -- a family is positive if ANY member is; it is a known negative only if EVERY member is a known negative
        SELECT detector, 'family', family_id, COUNT(*), MAX(score),
               BOOL_OR(is_pos_big_shift), BOOL_OR(is_pos_big_spike), BOOL_OR(is_pos_relative),
               all_neg AND BOOL_OR(is_neg_january_list_increase),
               all_neg AND BOOL_OR(is_neg_normal_move),
               all_neg AND BOOL_OR(is_neg_hard_january),
               all_neg AND BOOL_OR(is_neg_hard_volatile)
        FROM (SELECT *, BOOL_AND(is_neg_january_list_increase OR is_neg_normal_move OR is_neg_hard_january
                                 OR is_neg_hard_volatile) OVER (PARTITION BY detector, family_id) AS all_neg
              FROM scored WHERE split = 'test')
        GROUP BY detector, family_id, all_neg)
    SELECT *, ROW_NUMBER() OVER (PARTITION BY detector, unit ORDER BY score DESC NULLS LAST, XXHASH64(unit_id)) AS rnk
    FROM units
""").cache().createOrReplaceTempView("units")

header("test split: how many units of each label")
q("""SELECT unit, COUNT(*) AS units, MAX(size) AS largest, SUM(CAST(pos_shift AS INT)) AS big_shift,
            SUM(CAST(pos_rel AS INT)) AS relative, SUM(CAST(pos_spike AS INT)) AS spike,
            SUM(CAST(neg_jan AS INT)) AS neg_january, SUM(CAST(neg_normal AS INT)) AS neg_normal,
            SUM(CAST(hard_jan AS INT)) AS hard_january, SUM(CAST(hard_vol AS INT)) AS hard_volatile
     FROM units WHERE detector = 'baseline_abs_pct' GROUP BY unit ORDER BY unit DESC""")

header("test split: family sizes, and the largest families")
q("""WITH f AS (SELECT family_id, COUNT(*) AS n, MIN(published_week) AS week,
                       CONCAT_WS(' / ', SLICE(ARRAY_SORT(COLLECT_SET(REGEXP_EXTRACT(drug_group, '^([^ |-]+)', 1))), 1, 6)) AS names,
                       COUNT(DISTINCT REGEXP_EXTRACT(drug_group, '^([^ |-]+)', 1)) AS n_names
                FROM scored WHERE detector = 'baseline_abs_pct' AND split = 'test' GROUP BY family_id)
     SELECT CASE WHEN n = 1 THEN '1' WHEN n <= 3 THEN '2-3' WHEN n <= 10 THEN '4-10' ELSE '11+' END AS size,
            COUNT(*) AS families, SUM(n) AS events FROM f GROUP BY 1 ORDER BY MIN(n)""")
q("""SELECT family_id, n, n_names, names FROM (
         SELECT family_id, COUNT(*) AS n, COUNT(DISTINCT REGEXP_EXTRACT(drug_group, '^([^ |-]+)', 1)) AS n_names,
                CONCAT_WS(' / ', SLICE(ARRAY_SORT(COLLECT_SET(REGEXP_EXTRACT(drug_group, '^([^ |-]+)', 1))), 1, 6)) AS names
         FROM scored WHERE detector = 'baseline_abs_pct' AND split = 'test' GROUP BY family_id)
     ORDER BY n DESC LIMIT 5""")

header("test split: weeks where big moves (> 50%) form several families (what the family key does NOT merge)")
q("""SELECT published_week, COUNT(*) AS big_events, COUNT(DISTINCT family_id) AS families,
            CONCAT_WS(' / ', SLICE(ARRAY_SORT(COLLECT_SET(REGEXP_EXTRACT(drug_group, '^([^ |-]+)', 1))), 1, 8)) AS names
     FROM scored WHERE detector = 'baseline_abs_pct' AND split = 'test' AND ABS(pct_change) > 50
     GROUP BY published_week HAVING COUNT(DISTINCT family_id) >= 3 ORDER BY families DESC LIMIT 8""")

n_values = ", ".join(str(n) for n in TOP_N)
at_n = spark.sql(f"""
    SELECT u.unit, u.detector, t.n AS top_n,
        ROUND(SUM(CASE WHEN pos_shift AND rnk <= t.n THEN 1 ELSE 0 END) / SUM(CAST(pos_shift AS INT)), 3) AS recall_big_shift,
        ROUND(SUM(CASE WHEN pos_rel   AND rnk <= t.n THEN 1 ELSE 0 END) / SUM(CAST(pos_rel   AS INT)), 3) AS recall_relative,
        SUM(CASE WHEN hard_jan   AND rnk <= t.n THEN 1 ELSE 0 END)  AS hard_ff_january,
        SUM(CASE WHEN hard_vol   AND rnk <= t.n THEN 1 ELSE 0 END)  AS hard_ff_volatile,
        SUM(CASE WHEN neg_jan    AND rnk <= t.n THEN 1 ELSE 0 END)  AS ff_january_all,
        SUM(CASE WHEN neg_normal AND rnk <= t.n THEN 1 ELSE 0 END)  AS ff_normal,
        ROUND(SUM(CASE WHEN (pos_shift OR pos_spike OR pos_rel) AND rnk <= t.n THEN 1 ELSE 0 END) / t.n, 3)
                                                                    AS known_pos_share
    FROM units AS u CROSS JOIN (SELECT EXPLODE(ARRAY({n_values})) AS n) AS t
    GROUP BY u.unit, u.detector, t.n
    ORDER BY u.unit DESC, t.n, u.detector""")
header("test split: metrics in the top N flags (hard_ff = hard negatives wrongly flagged; ff_january_all includes hard)")
at_n.show(60, truncate=False)

# average precision: positives of one type against one set of negatives, in each detector's order
ap = spark.sql("""
    WITH typed AS (
        SELECT 'big_shift' AS pos_type, 'easy' AS negs, detector, unit, rnk, pos_shift AS is_pos FROM units
        WHERE pos_shift OR neg_jan OR neg_normal
        UNION ALL SELECT 'relative', 'easy', detector, unit, rnk, pos_rel FROM units
        WHERE pos_rel OR neg_jan OR neg_normal
        UNION ALL SELECT 'big_shift', 'hard', detector, unit, rnk, pos_shift FROM units
        WHERE pos_shift OR hard_jan OR hard_vol
        UNION ALL SELECT 'relative', 'hard', detector, unit, rnk, pos_rel FROM units
        WHERE pos_rel OR hard_jan OR hard_vol),
    cum AS (
        SELECT *, SUM(CAST(is_pos AS INT)) OVER (PARTITION BY pos_type, negs, detector, unit ORDER BY rnk) AS pos_so_far,
                  ROW_NUMBER()            OVER (PARTITION BY pos_type, negs, detector, unit ORDER BY rnk) AS pool_rank
        FROM typed)
    SELECT unit, detector,
        ROUND(AVG(CASE WHEN pos_type = 'big_shift' AND negs = 'easy' AND is_pos THEN pos_so_far / pool_rank END), 3) AS ap_shift_easy,
        ROUND(AVG(CASE WHEN pos_type = 'relative'  AND negs = 'easy' AND is_pos THEN pos_so_far / pool_rank END), 3) AS ap_rel_easy,
        ROUND(AVG(CASE WHEN pos_type = 'big_shift' AND negs = 'hard' AND is_pos THEN pos_so_far / pool_rank END), 3) AS ap_shift_hard,
        ROUND(AVG(CASE WHEN pos_type = 'relative'  AND negs = 'hard' AND is_pos THEN pos_so_far / pool_rank END), 3) AS ap_rel_hard
    FROM cum GROUP BY unit, detector ORDER BY unit DESC, detector""")
header("test split: average precision vs easy negatives and vs HARD negatives (1.0 = all positives first)")
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
pivot = ",\n".join(f"MAX(CASE WHEN detector = '{d}' THEN top_pct_all END) AS `{SHORT.get(d, d)}`" for d in DETECTORS)
passes = ",\n".join(
    f"""MAX(CASE WHEN detector = '{d}' THEN CASE WHEN test_case_expect = 'flag' THEN top_pct_all <= 1
                                           ELSE top_pct_all > 10 END END) AS `ok_{SHORT.get(d, d)}`"""
    for d in DETECTORS)
q(f"""SELECT test_case, test_case_expect AS expect, published_week, ROUND(pct_change, 1) AS pct, {pivot}, {passes}
      FROM scored WHERE test_case_expect IS NOT NULL
      GROUP BY test_case, test_case_expect, published_week, pct_change ORDER BY test_case, published_week""")

for top_det in TOP_FAMILIES_FOR:
    header(f"top 20 families, test split: {top_det} (with the same family's rank under other detectors)")
    other_ranks = ", ".join(f"MAX(CASE WHEN u.detector = '{d}' THEN u.rnk END) AS `rank_{SHORT.get(d, d)}`"
                            for d in DETECTORS if d != top_det)
    q(f"""WITH top AS (SELECT unit_id, rnk FROM units WHERE detector = '{top_det}' AND unit = 'family' AND rnk <= 20),
          members AS (
              SELECT s.family_id, COUNT(*) AS n,
                     MIN(s.published_week) AS week,
                     MAX_BY(s.drug_group, s.score)                   AS top_member,
                     MAX_BY(ROUND(s.pct_change, 1), s.score)         AS top_pct,
                     CONCAT_WS(',', ARRAY_SORT(COLLECT_SET(s.category))) AS categories
              FROM scored s WHERE s.detector = '{top_det}' AND s.split = 'test' GROUP BY s.family_id)
          SELECT t.rnk AS rank, m.week, m.n, m.top_member, m.top_pct, m.categories, {other_ranks}
          FROM top t JOIN members m ON m.family_id = t.unit_id
          JOIN units u ON u.unit_id = t.unit_id AND u.unit = 'family'
          GROUP BY t.rnk, m.week, m.n, m.top_member, m.top_pct, m.categories ORDER BY t.rnk""")

os.makedirs("reports", exist_ok=True)
with open(f"reports/metrics_{RUN_NAME}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split", "unit", "detector", "metric", "value"])
    for r in at_n.collect():
        for m in ["recall_big_shift", "recall_relative", "hard_ff_january", "hard_ff_volatile",
                  "ff_january_all", "ff_normal", "known_pos_share"]:
            w.writerow(["test", r["unit"], r["detector"], f"{m}@{r['top_n']}", r[m]])
    for r in ap.collect():
        for m in ["ap_shift_easy", "ap_rel_easy", "ap_shift_hard", "ap_rel_hard"]:
            w.writerow(["test", r["unit"], r["detector"], m, r[m]])
    for r in spikes.collect():
        for m in ["median_top_pct", "in_top_1pct", "in_top_5pct"]:
            w.writerow(["all", "event", r["detector"], f"spike_{m}", r[m]])
print(f"\nsaved reports/metrics_{RUN_NAME}.csv")
spark.stop()
