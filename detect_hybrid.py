"""Step 8b-2: hybrid score = "big in absolute terms" AND "unusual for this drug".

Two ingredients per real event (both from data/eval/scores_zscore, past-only):
    size = |ln(new/old)|               (how big the move is)
    rel  = |z| of variant B (seasonal)  (how unusual it is for this drug)
Each is turned into a percentile against a FIXED reference: all events published before 2025 (the train
period). So a test event is only compared with the past, and the two very different scales (a log change
vs a z in the hundreds) become comparable.

    hybrid = (percentile(size) + percentile(rel)) / 2       -> high only when BOTH are high

Above the reference maximum every event gets percentile 1.0, so a tiny tie-breaker (< 1e-6, far smaller than
one percentile step) keeps those events ordered by size x rel. Output: data/eval/scores_hybrid/.
"""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

REFERENCE_END = "2025-01-01"   # reference distribution = events published before this date

spark = (SparkSession.builder.master("local[*]").appName("detect_hybrid")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.read.parquet("data/eval/scores_zscore").createOrReplaceTempView("zs")


def percentile_vs_reference(value_sql, name):
    """Share of reference (train-period) values <= this event's value. Reference rows sort before events on ties."""
    spark.sql(f"""
        WITH vals AS (
            SELECT drug_group, published_week, {value_sql} AS v, 0 AS is_ref FROM zs WHERE {value_sql} IS NOT NULL
            UNION ALL
            SELECT NULL, NULL, {value_sql}, 1 FROM zs
            WHERE {value_sql} IS NOT NULL AND published_week < DATE'{REFERENCE_END}'),
        cum AS (
            SELECT *, SUM(is_ref) OVER (ORDER BY v, is_ref DESC ROWS UNBOUNDED PRECEDING) AS refs_at_or_below,
                      SUM(is_ref) OVER ()                                                  AS n_ref
            FROM vals)
        SELECT drug_group, published_week, v AS {name}, refs_at_or_below / n_ref AS pct_{name}
        FROM cum WHERE is_ref = 0
    """).createOrReplaceTempView(f"p_{name}")


percentile_vs_reference("ABS(x)", "size")
percentile_vs_reference("score_b", "rel")

spark.sql("""
    SELECT s.drug_group, s.published_week,
           ROUND(s.size, 6) AS size, ROUND(s.pct_size, 6) AS pct_size,
           ROUND(r.rel, 3)  AS rel,  ROUND(r.pct_rel, 6)  AS pct_rel,
           (s.pct_size + r.pct_rel) / 2 + 1e-7 * LN(1 + s.size * r.rel) AS score_hybrid
    FROM p_size AS s
    JOIN p_rel  AS r USING (drug_group, published_week)
""").write.mode("overwrite").parquet("data/eval/scores_hybrid")

spark.read.parquet("data/eval/scores_hybrid").createOrReplaceTempView("h")
print("\n======== scores_hybrid summary ========")
spark.sql(f"""
    SELECT CASE WHEN published_week < DATE'{REFERENCE_END}' THEN 'train (reference)' ELSE 'test' END AS period,
           COUNT(*) AS events,
           SUM(CASE WHEN pct_size >= 1 THEN 1 ELSE 0 END) AS size_above_reference_max,
           SUM(CASE WHEN pct_rel  >= 1 THEN 1 ELSE 0 END) AS rel_above_reference_max,
           ROUND(PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY score_hybrid), 3) AS median_hybrid
    FROM h GROUP BY 1 ORDER BY 1""").show(truncate=False)
spark.stop()
