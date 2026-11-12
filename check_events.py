"""Step 6b: checks on data/tables/price_events."""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

spark = (SparkSession.builder.master("local[*]").appName("check_events")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
for name in ["ndc_dim", "group_dim", "group_weekly", "price_events"]:
    spark.read.parquet(f"data/tables/{name}").createOrReplaceTempView(name)


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def q(sql, n=40):
    spark.sql(sql).show(n, truncate=False)


EVENT_COLS = """published_week, old_price, new_price, pct_change, effective_date, publication_lag_days,
                explanation_codes, is_january, mix_only, after_gap,
                outcome_price_4w, outcome_price_8w, outcome_reverted, outcome_persisted"""

header("completeness: one event per price_changed week")
q("""SELECT (SELECT COUNT(*) FROM group_weekly WHERE price_changed) AS changed_weeks_in_group_weekly,
            (SELECT COUNT(*) FROM price_events)                     AS events,
            (SELECT COUNT(*) - COUNT(DISTINCT drug_group, published_week) FROM price_events) AS duplicate_events""")

header("events overall and per year")
q("""SELECT COALESCE(CAST(YEAR(published_week) AS STRING), 'ALL') AS year,
            COUNT(*)                                                AS events,
            ROUND(AVG(CAST(mix_only  AS INT)) * 100, 1)             AS pct_mix_only,
            ROUND(AVG(CAST(after_gap AS INT)) * 100, 1)             AS pct_after_gap,
            ROUND(AVG(CASE WHEN direction = 'up' THEN 1 ELSE 0 END) * 100, 1) AS pct_up
     FROM price_events GROUP BY ROLLUP(YEAR(published_week)) ORDER BY year""")

header("events per group")
q("""WITH per_group AS (
         SELECT g.drug_group, COUNT(e.drug_group) AS n FROM group_dim g
         LEFT JOIN price_events e ON e.drug_group = g.drug_group GROUP BY g.drug_group)
     SELECT CASE WHEN n = 0 THEN '0' WHEN n = 1 THEN '1' WHEN n <= 5 THEN '2-5' WHEN n <= 20 THEN '6-20'
                 WHEN n <= 50 THEN '21-50' ELSE '51+' END AS events_per_group,
            COUNT(*) AS groups
     FROM per_group GROUP BY 1 ORDER BY MIN(n)""")

header("brand vs generic")
q("""SELECT CASE WHEN is_brand THEN 'brand' ELSE 'generic' END AS kind,
            COUNT(*)                                                       AS events,
            COUNT(DISTINCT drug_group)                                     AS groups_with_events,
            ROUND(COUNT(*) / COUNT(DISTINCT drug_group), 1)                AS events_per_group,
            PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY ABS(pct_change))   AS median_abs_pct,
            ROUND(AVG(CASE WHEN direction = 'up' THEN 1 ELSE 0 END) * 100, 1) AS pct_up,
            ROUND(AVG(CAST(is_january AS INT)) * 100, 1)                   AS pct_january,
            ROUND(AVG(CAST(mix_only AS INT)) * 100, 1)                     AS pct_mix_only
     FROM price_events GROUP BY 1""")
print("brand increases >= 1%: share effective in January")
q("""SELECT CASE WHEN is_brand THEN 'brand' ELSE 'generic' END AS kind, COUNT(*) AS increases_over_1pct,
            ROUND(AVG(CAST(is_january AS INT)) * 100, 1) AS pct_january
     FROM price_events WHERE pct_change >= 1 AND NOT mix_only GROUP BY 1""")

header("Otezla 30 mg: all events")
q(f"""SELECT event_number, weeks_since_last_event, {EVENT_COLS}
      FROM price_events e JOIN ndc_dim d ON d.drug_group = e.drug_group
      WHERE d.ndc = '55513013760' ORDER BY published_week""")

header("level shifts: isosorbide mononitrate 20 mg, isoniazid 300 mg, prucalopride 2 mg (|pct_change| > 50%)")
q(f"""SELECT drug_group, {EVENT_COLS}, outcome_price_12w
      FROM price_events
      WHERE (drug_group LIKE 'ISOSORBIDE MONONIT 20 MG TAB|%' OR drug_group LIKE 'ISONIAZID 300 MG TABLET|%'
             OR drug_group LIKE 'PRUCALOPRIDE 2 MG TABLET|%')
        AND ABS(pct_change) > 50
      ORDER BY drug_group, published_week""")

header("big moves (|pct_change| > 50%): reverted vs persisted")
q("""SELECT mix_only,
            COUNT(*)                                                                         AS big_moves,
            SUM(CASE WHEN outcome_reverted THEN 1 ELSE 0 END)                                AS reverted,
            SUM(CASE WHEN outcome_persisted THEN 1 ELSE 0 END)                               AS persisted,
            SUM(CASE WHEN outcome_reverted AND outcome_persisted THEN 1 ELSE 0 END)          AS both,
            SUM(CASE WHEN NOT outcome_reverted AND NOT outcome_persisted THEN 1 ELSE 0 END)  AS neither,
            SUM(CASE WHEN outcome_reverted IS NULL OR outcome_persisted IS NULL THEN 1 ELSE 0 END) AS unknown_yet
     FROM price_events WHERE ABS(pct_change) > 50 GROUP BY mix_only""")
for label, cond in [("reverted (spike that went back)", "outcome_reverted AND NOT COALESCE(outcome_persisted, FALSE)"),
                    ("persisted (level shift)", "outcome_persisted AND NOT outcome_reverted")]:
    print(f"3 examples - {label}, real price moves only (mix_only = false):")
    q(f"""SELECT drug_group, published_week, old_price, new_price, pct_change, n_ndcs_priced,
                 ndc_confirmed_share, outcome_price_4w, outcome_price_8w
          FROM price_events WHERE ABS(pct_change) > 50 AND NOT mix_only AND {cond}
          ORDER BY ABS(pct_change) DESC LIMIT 3""")

header("publication lag: published_week - effective_date (days)")
q("""SELECT CASE WHEN publication_lag_days < 0 THEN 'a: negative (effective AFTER published)'
                 WHEN publication_lag_days = 0 THEN 'b: 0'
                 WHEN publication_lag_days <= 7 THEN 'c: 1-7'
                 WHEN publication_lag_days <= 14 THEN 'd: 8-14'
                 WHEN publication_lag_days <= 28 THEN 'e: 15-28'
                 WHEN publication_lag_days <= 90 THEN 'f: 29-90'
                 ELSE 'g: over 90' END AS lag_days,
            COUNT(*) AS events,
            ROUND(COUNT(*) * 100 / SUM(COUNT(*)) OVER (), 1) AS pct
     FROM price_events WHERE NOT mix_only GROUP BY 1 ORDER BY 1""")
q("""SELECT CASE WHEN is_brand THEN 'brand' ELSE 'generic' END AS kind,
            PERCENTILE_DISC(0.5)  WITHIN GROUP (ORDER BY publication_lag_days) AS p50,
            PERCENTILE_DISC(0.9)  WITHIN GROUP (ORDER BY publication_lag_days) AS p90,
            PERCENTILE_DISC(0.99) WITHIN GROUP (ORDER BY publication_lag_days) AS p99,
            MAX(publication_lag_days) AS max_days
     FROM price_events WHERE NOT mix_only GROUP BY 1""")
print("Longest lags:")
q("""SELECT drug_group, published_week, effective_date, publication_lag_days, pct_change, explanation_codes
     FROM price_events WHERE NOT mix_only ORDER BY publication_lag_days DESC LIMIT 5""")

spark.stop()
