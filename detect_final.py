"""Step 8d: final detector = hybrid score (main) + IF-small (second opinion) + caution flags.

Inputs (all past-only / publication-time): data/eval/scores_hybrid, data/eval/scores_iforest, data/tables/price_events.
No eval labels and no outcome_* columns are read.

  - "top 1%" of a detector = score above that detector's 99th percentile on the TRAIN events (published before
    2025). A fixed cut-off: a quiet week can't push ordinary moves into its own "top 1%".
    (IF-small's train scores are in-sample: the forest was fitted on those events. Fine for a cut-off.)
  - confidence: high = both in their top 1%; medium = only one (medium_source says which); none = neither.
  - score_final = 2 * tier + score_hybrid  (tier: high 2, medium 1, none 0; score_hybrid is in [0, 1 + tiny]),
    i.e. rank by confidence first, then by the hybrid.
  - pool: real events only (mix_only and artefact_suspect are already left out of scores_hybrid).
  - caution flags (shown, not excluded): class_changed, after_gap (+ how many weeks were missing),
    brand increase effective in January, round (expected list-price increase) or not round (Tekturna pattern).

Output: data/eval/scores_final/.
"""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

REFERENCE_END = "2025-01-01"
TOP_SHARE = 0.01

spark = (SparkSession.builder.master("local[*]").appName("detect_final")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.read.parquet("data/eval/scores_hybrid").createOrReplaceTempView("hy")
spark.read.parquet("data/eval/scores_iforest").createOrReplaceTempView("iforest")
spark.read.parquet("data/tables/price_events").createOrReplaceTempView("pe")

spark.sql("""
    SELECT h.drug_group, h.published_week, h.score_hybrid, h.pct_size, i.score_if_small,
           i.typical_past_move, i.typical_from_class
    FROM hy AS h JOIN iforest AS i USING (drug_group, published_week)
""").createOrReplaceTempView("both")

cut = spark.sql(f"""
    SELECT PERCENTILE_DISC({1 - TOP_SHARE}) WITHIN GROUP (ORDER BY score_hybrid)   AS cut_hybrid,
           PERCENTILE_DISC({1 - TOP_SHARE}) WITHIN GROUP (ORDER BY score_if_small) AS cut_if_small,
           COUNT(*) AS n_train
    FROM both WHERE published_week < DATE'{REFERENCE_END}'""").first()
print(f"train events: {cut.n_train:,}; top-1% cut-offs: hybrid > {cut.cut_hybrid:.6f}, "
      f"IF-small > {cut.cut_if_small:.6f}")

spark.sql(f"""
    WITH flagged AS (
        SELECT b.*,
               b.score_hybrid   > {cut.cut_hybrid}   AS hybrid_top1,
               b.score_if_small > {cut.cut_if_small} AS if_small_top1
        FROM both AS b)
    SELECT f.drug_group, f.published_week,
           f.score_hybrid, f.score_if_small, f.pct_size, f.typical_past_move, f.typical_from_class,
           f.hybrid_top1, f.if_small_top1,
           CASE WHEN f.hybrid_top1 AND f.if_small_top1 THEN 'high'
                WHEN f.hybrid_top1 OR  f.if_small_top1 THEN 'medium' ELSE 'none' END AS confidence,
           CASE WHEN f.hybrid_top1 AND NOT f.if_small_top1 THEN 'hybrid_only'
                WHEN f.if_small_top1 AND NOT f.hybrid_top1 THEN 'if_only' END       AS medium_source,
           -- cautions: publication-time data only
           p.class_changed                                                AS caution_class_changed,
           p.after_gap                                                    AS caution_after_gap,
           IF(p.after_gap, CAST(DATEDIFF(p.published_week, p.prev_published_week) / 7 AS INT) - 1, 0)
                                                                          AS gap_weeks_missing,
           p.is_brand AND p.is_january AND p.pct_change > 0
             AND ABS(p.pct_change - ROUND(p.pct_change)) <= 0.1           AS caution_jan_brand_round,
           p.is_brand AND p.is_january AND p.pct_change > 0
             AND ABS(p.pct_change - ROUND(p.pct_change)) > 0.1            AS caution_jan_brand_not_round
    FROM flagged AS f
    JOIN pe AS p USING (drug_group, published_week)
""").createOrReplaceTempView("fin")

spark.sql("""
    SELECT *, CASE confidence WHEN 'high' THEN 2 WHEN 'medium' THEN 1 ELSE 0 END * 2 + score_hybrid AS score_final
    FROM fin""").write.mode("overwrite").parquet("data/eval/scores_final")

spark.read.parquet("data/eval/scores_final").createOrReplaceTempView("sf")
print("\n======== scores_final: confidence by period ========")
spark.sql(f"""
    SELECT CASE WHEN published_week < DATE'{REFERENCE_END}' THEN 'train' ELSE 'test' END AS period,
           confidence, COALESCE(medium_source, '') AS source, COUNT(*) AS events,
           ROUND(COUNT(*) * 100 / SUM(COUNT(*)) OVER (PARTITION BY
                 CASE WHEN published_week < DATE'{REFERENCE_END}' THEN 'train' ELSE 'test' END), 2) AS pct_of_period,
           SUM(CAST(caution_class_changed AS INT)) AS c_class, SUM(CAST(caution_after_gap AS INT)) AS c_gap,
           SUM(CAST(caution_jan_brand_round AS INT)) AS c_jan_round,
           SUM(CAST(caution_jan_brand_not_round AS INT)) AS c_jan_not_round
    FROM sf GROUP BY 1, 2, 3 ORDER BY 1 DESC, 2, 3""").show(truncate=False)
spark.stop()
