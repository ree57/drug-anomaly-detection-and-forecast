"""Step 8b: Method 1, rolling robust z-score detector, in two variants.

For every real price event (mix_only and artefact_suspect excluded), x = ln(new_price / old_price) is compared with the
same group's PREVIOUS real events only:

    robust z = (x - median) / max(1.4826 * MAD, SCALE_FLOOR)        score = |z|

Variant A (plain):    history = the group's previous 12 real events (needs >= 6).
Variant B (seasonal): brand events effective in January -> only the group's previous January events
                      (needs >= 3); other brand events -> only previous non-January events (needs >= 6);
                      generics -> same as A.
Too little history -> compare with the class distribution instead (all real events of that class published
before the start of this event's month) and set fallback_*. No class history at all -> score NULL.

Inputs: data/tables/price_events (no outcome columns are read). Output: data/eval/scores_zscore/.
"""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

HISTORY = 12           # how many previous real events of the group to look at
MIN_HISTORY = 6        # fewer than this -> class fallback
MIN_JAN_HISTORY = 3    # variant B, brand January events: at most 4 earlier Januaries exist in 2022-2026
MAD_TO_SD = 1.4826     # makes MAD comparable to a standard deviation
SCALE_FLOOR = 0.01     # spread never below ~1%, so a drug that never moved can't give infinite z

spark = (SparkSession.builder.master("local[*]").appName("detect_zscore")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.read.parquet("data/tables/price_events").createOrReplaceTempView("price_events")


def median_of(arr):
    """SQL expression: median of a non-empty array (average of the two middle values when the size is even)."""
    s = f"ARRAY_SORT({arr})"
    lo = f"ELEMENT_AT({s}, CAST(FLOOR((SIZE({arr}) + 1) / 2) AS INT))"
    hi = f"ELEMENT_AT({s}, CAST(CEIL((SIZE({arr}) + 1) / 2) AS INT))"
    return f"CASE WHEN SIZE({arr}) > 0 THEN ({lo} + {hi}) / 2 END"


# 1. real events, with only what is known at publication time
spark.sql("""
    SELECT drug_group, published_week, is_brand,
           LN(new_price / old_price)                                   AS x,
           TRUNC(published_week, 'MM')                                 AS month,
           CASE WHEN is_brand THEN 'brand' ELSE 'generic' END          AS plain_class,
           CASE WHEN NOT is_brand THEN 'generic'
                WHEN is_january THEN 'brand_jan' ELSE 'brand_other' END AS season_class
    FROM price_events
    WHERE NOT mix_only AND NOT artefact_suspect   -- both known at publication: not real price changes
""").createOrReplaceTempView("real_events")

# 2. each event's own past: the frame ends 1 row BEFORE the event, so the event never sees itself
spark.sql(f"""
    SELECT *,
        COLLECT_LIST(x) OVER (PARTITION BY drug_group ORDER BY published_week
                              ROWS BETWEEN {HISTORY} PRECEDING AND 1 PRECEDING) AS hist_a,
        COLLECT_LIST(x) OVER (PARTITION BY drug_group, season_class ORDER BY published_week
                              ROWS BETWEEN {HISTORY} PRECEDING AND 1 PRECEDING) AS hist_b
    FROM real_events
""").createOrReplaceTempView("hist")
spark.sql(f"SELECT *, {median_of('hist_a')} AS med_a, {median_of('hist_b')} AS med_b FROM hist") \
    .createOrReplaceTempView("hist_med")
spark.sql(f"""
    SELECT *,
        {median_of('TRANSFORM(hist_a, v -> ABS(v - med_a))')} AS mad_a,
        {median_of('TRANSFORM(hist_b, v -> ABS(v - med_b))')} AS mad_b
    FROM hist_med
""").createOrReplaceTempView("group_stats")

# 3. class fallback: for every month, all real events of the class published BEFORE that month began
spark.sql("""
    SELECT m.month, r.plain_class, r.season_class, r.x
    FROM (SELECT DISTINCT month FROM real_events) AS m
    JOIN real_events AS r ON r.published_week < m.month
""").cache().createOrReplaceTempView("class_long")
spark.sql("""
    WITH long AS (
        SELECT 'A' AS variant, month, plain_class  AS cls, x FROM class_long
        UNION ALL
        SELECT 'B' AS variant, month, season_class AS cls, x FROM class_long),
    med AS (SELECT variant, month, cls, MEDIAN(x) AS med FROM long GROUP BY 1, 2, 3)
    SELECT l.variant, l.month, l.cls, m.med, MEDIAN(ABS(l.x - m.med)) AS mad
    FROM long AS l JOIN med AS m USING (variant, month, cls)
    GROUP BY l.variant, l.month, l.cls, m.med
""").createOrReplaceTempView("class_stats")

# 4. pick group or class stats, guard the spread, compute z
spark.sql(f"""
    WITH chosen AS (
        SELECT g.drug_group, g.published_week, g.season_class, g.x,
            SIZE(g.hist_a)                                   AS n_hist_a,
            SIZE(g.hist_a) < {MIN_HISTORY}                   AS fallback_a,
            IF(SIZE(g.hist_a) < {MIN_HISTORY}, ca.med, g.med_a)                AS center_a,
            {MAD_TO_SD} * IF(SIZE(g.hist_a) < {MIN_HISTORY}, ca.mad, g.mad_a)  AS raw_scale_a,
            SIZE(g.hist_b)                                   AS n_hist_b,
            SIZE(g.hist_b) < IF(g.season_class = 'brand_jan', {MIN_JAN_HISTORY}, {MIN_HISTORY}) AS fallback_b,
            IF(SIZE(g.hist_b) < IF(g.season_class = 'brand_jan', {MIN_JAN_HISTORY}, {MIN_HISTORY}),
               cb.med, g.med_b)                              AS center_b,
            {MAD_TO_SD} * IF(SIZE(g.hist_b) < IF(g.season_class = 'brand_jan', {MIN_JAN_HISTORY}, {MIN_HISTORY}),
               cb.mad, g.mad_b)                              AS raw_scale_b
        FROM group_stats AS g
        LEFT JOIN class_stats AS ca ON ca.variant = 'A' AND ca.month = g.month AND ca.cls = g.plain_class
        LEFT JOIN class_stats AS cb ON cb.variant = 'B' AND cb.month = g.month AND cb.cls = g.season_class)
    SELECT drug_group, published_week, season_class, ROUND(x, 6) AS x,
        n_hist_a, fallback_a, ROUND(center_a, 6) AS center_a,
        ROUND(GREATEST(raw_scale_a, {SCALE_FLOOR}), 6)                    AS scale_a,
        raw_scale_a < {SCALE_FLOOR}                                       AS scale_floored_a,
        ROUND((x - center_a) / GREATEST(raw_scale_a, {SCALE_FLOOR}), 3)   AS z_a,
        ROUND(ABS(x - center_a) / GREATEST(raw_scale_a, {SCALE_FLOOR}), 3) AS score_a,
        n_hist_b, fallback_b, ROUND(center_b, 6) AS center_b,
        ROUND(GREATEST(raw_scale_b, {SCALE_FLOOR}), 6)                    AS scale_b,
        raw_scale_b < {SCALE_FLOOR}                                       AS scale_floored_b,
        ROUND((x - center_b) / GREATEST(raw_scale_b, {SCALE_FLOOR}), 3)   AS z_b,
        ROUND(ABS(x - center_b) / GREATEST(raw_scale_b, {SCALE_FLOOR}), 3) AS score_b
    FROM chosen
""").write.mode("overwrite").parquet("data/eval/scores_zscore")

spark.read.parquet("data/eval/scores_zscore").createOrReplaceTempView("zs")
print("\n======== scores_zscore summary ========")
spark.sql("""
    SELECT COUNT(*) AS scored_events,
           SUM(CASE WHEN z_a IS NULL THEN 1 ELSE 0 END)                     AS no_history,
           ROUND(AVG(CAST(fallback_a AS INT)) * 100, 1)                     AS pct_fallback_a,
           ROUND(AVG(CAST(fallback_b AS INT)) * 100, 1)                     AS pct_fallback_b,
           ROUND(AVG(CAST(scale_floored_a AS INT)) * 100, 1)                AS pct_floored_a,
           ROUND(AVG(CAST(scale_floored_b AS INT)) * 100, 1)                AS pct_floored_b,
           ROUND(AVG(CASE WHEN score_a > 3 THEN 1 ELSE 0 END) * 100, 1)     AS pct_abs_z_a_over_3,
           ROUND(AVG(CASE WHEN score_b > 3 THEN 1 ELSE 0 END) * 100, 1)     AS pct_abs_z_b_over_3
    FROM zs""").show(truncate=False)
print("by season_class (variant B's three routes)")
spark.sql("""
    SELECT season_class, COUNT(*) AS events,
           ROUND(AVG(CAST(fallback_b AS INT)) * 100, 1)     AS pct_fallback_b,
           PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY score_a) AS median_score_a,
           PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY score_b) AS median_score_b
    FROM zs GROUP BY 1 ORDER BY 1""").show(truncate=False)
spark.stop()
