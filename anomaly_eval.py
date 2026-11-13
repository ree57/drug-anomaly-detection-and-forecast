"""Step 8a: build the anomaly evaluation set from price_events.

Runs sql/eval_events.sql and writes data/eval/eval_events/ (Parquet), then prints counts per category,
examples, and the named test cases. See reports/anomaly_eval_plan.md for the definitions and metrics.
"""
import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

spark = (SparkSession.builder.master("local[*]").appName("anomaly_eval")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.read.parquet("data/tables/price_events").createOrReplaceTempView("price_events")

query = Path("sql/eval_events.sql").read_text(encoding="utf-8")
spark.sql(query).write.mode("overwrite").parquet("data/eval/eval_events")
spark.read.parquet("data/eval/eval_events").createOrReplaceTempView("ev")


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def q(sql, n=40):
    spark.sql(sql).show(n, truncate=False)


header("reconcile with price_events")
q("""SELECT (SELECT COUNT(*) FROM price_events) AS price_events, (SELECT COUNT(*) FROM ev) AS eval_events""")

header("primary category (one per event), overall and by split")
q("""SELECT category, COUNT(*) AS events, ROUND(COUNT(*) * 100 / SUM(COUNT(*)) OVER (), 2) AS pct,
            SUM(CASE WHEN split = 'train' THEN 1 ELSE 0 END) AS train_2022_24,
            SUM(CASE WHEN split = 'test'  THEN 1 ELSE 0 END) AS test_2025_26,
            SUM(CASE WHEN after_gap THEN 1 ELSE 0 END)       AS after_gap
     FROM ev GROUP BY category ORDER BY category""")

header("label flags (an event can carry more than one) and overlaps")
q("""SELECT SUM(CAST(is_pos_big_spike AS INT))            AS pos_big_spike,
            SUM(CAST(is_pos_big_shift AS INT))            AS pos_big_shift,
            SUM(CAST(is_pos_relative AS INT))             AS pos_relative,
            SUM(CASE WHEN is_pos_relative AND is_pos_big_shift THEN 1 ELSE 0 END) AS relative_and_big_shift,
            SUM(CASE WHEN is_pos_relative AND is_pos_big_spike THEN 1 ELSE 0 END) AS relative_and_big_spike,
            SUM(CAST(is_neg_january_list_increase AS INT)) AS neg_january,
            SUM(CAST(is_neg_normal_move AS INT))           AS neg_normal,
            SUM(CASE WHEN is_neg_january_list_increase AND is_neg_normal_move THEN 1 ELSE 0 END) AS neg_both
     FROM ev""")

header("pos_relative: who are they?")
q("""SELECT CASE WHEN NOT is_brand THEN 'generic'
                 WHEN is_january THEN 'brand, January (decrease)'
                 WHEN direction = 'up' THEN 'brand, OFF-CYCLE increase'
                 ELSE 'brand, decrease' END AS kind,
            COUNT(*) AS events, COUNT(DISTINCT drug_group) AS groups,
            SUM(CASE WHEN ABS(pct_change) > 50 THEN 1 ELSE 0 END) AS also_over_50pct,
            PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY ABS(pct_change)) AS median_abs_pct,
            SUM(CASE WHEN split = 'test' THEN 1 ELSE 0 END) AS in_test_split
     FROM ev WHERE is_pos_relative GROUP BY 1 ORDER BY 1""")

EX_COLS = """drug_group, published_week, old_price, new_price, pct_change, prior_real_changes,
             prior_p95_abs_pct, outcome_price_8w, outcome_reverted, after_gap"""
for cat in ["pos_big_shift", "pos_big_spike", "pos_relative", "big_unclear",
            "neg_hard_january", "neg_hard_volatile", "neg_january_list_increase", "neg_normal_move",
            "excluded_mix_only", "excluded_artefact", "unlabelled"]:
    header(f"3 examples: {cat}")
    # hash order = a fixed but arbitrary pick, so examples aren't only the most extreme ones
    q(f"""SELECT {EX_COLS} FROM ev WHERE category = '{cat}'
          ORDER BY XXHASH64(drug_group, published_week) LIMIT 3""")

header("pos_relative: off-cycle brand increases (largest 5)")
q(f"""SELECT {EX_COLS} FROM ev
      WHERE is_pos_relative AND is_brand AND NOT is_january AND direction = 'up'
      ORDER BY ABS(pct_change) DESC LIMIT 5""")

header("artefact flags: big real moves (|pct_change| > 50%, not mix_only)")
q("""SELECT COUNT(*) AS big_moves,
            SUM(CAST(artefact_suspect AS INT))                                   AS artefact_suspect,
            SUM(CAST(class_changed AS INT))                                      AS class_changed,
            SUM(CASE WHEN artefact_suspect AND class_changed THEN 1 ELSE 0 END)  AS both,
            SUM(CASE WHEN is_artefact AND NOT artefact_suspect THEN 1 ELSE 0 END) AS manual_only,
            SUM(CASE WHEN artefact_suspect AND split = 'test' THEN 1 ELSE 0 END) AS suspect_in_test
     FROM ev WHERE ABS(pct_change) > 50 AND NOT mix_only""")
print("all events: artefact_suspect and class_changed")
q("""SELECT SUM(CAST(artefact_suspect AS INT)) AS artefact_suspect, SUM(CAST(class_changed AS INT)) AS class_changed,
            SUM(CASE WHEN class_changed AND ABS(pct_change) > 30 THEN 1 ELSE 0 END) AS class_changed_over_30pct
     FROM ev WHERE NOT mix_only""")
print("the three reviewed events")
q("""SELECT drug_group, published_week, ROUND(pct_change, 1) AS pct, matched_ndc_median_change, artefact_suspect,
            class_changed, category, artefact_reason
     FROM ev WHERE (drug_group LIKE 'DILANTIN 100 MG CAPSULE|%' AND published_week = DATE'2025-03-19')
                OR (drug_group LIKE 'SODIUM CHLORIDE 0.9% SOLUTION|ML|%' AND ABS(pct_change) > 50)
                OR (drug_group LIKE 'CLEMASTINE FUM 2.68 MG TABLET|%' AND published_week = DATE'2026-01-21')""")
print("largest 10 artefact_suspect big moves")
q("""SELECT drug_group, published_week, old_price, new_price, ROUND(pct_change, 1) AS pct, matched_ndc_median_change,
            class_changed, category
     FROM ev WHERE artefact_suspect AND ABS(pct_change) > 50 ORDER BY ABS(pct_change) DESC LIMIT 10""")

header("named test cases")
q(f"""SELECT test_case, test_case_expect, category, published_week, old_price, new_price, pct_change,
             prior_real_changes, prior_p95_abs_pct, split
      FROM ev WHERE test_case_expect IS NOT NULL ORDER BY test_case, published_week""")

spark.stop()
