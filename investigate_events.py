"""Top-20 review: look inside single flagged events to tell a real market move from a data quirk.

For each case: the event row, the group's weekly series around it, every NDC's own prices around it
(with explanation codes), and any change of description / unit / class for those NDCs.
"""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

CASES = [  # (drug_group, published_week of the flagged event)
    ("DILANTIN 100 MG CAPSULE|EA|B-ANDA", "2025-03-19"),
    ("CLEMASTINE FUM 2.68 MG TABLET|EA|G", "2026-01-21"),
    ("SODIUM CHLORIDE 0.9% SOLUTION|ML|G", "2026-07-22"),
]

spark = (SparkSession.builder.master("local[*]").appName("investigate_events")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.read.parquet("data/clean/nadac").createOrReplaceTempView("nadac")
for name in ["ndc_dim", "group_weekly", "price_events"]:
    spark.read.parquet(f"data/tables/{name}").createOrReplaceTempView(name)


def q(sql, n=80):
    spark.sql(sql).show(n, truncate=False)


for group, week in CASES:
    print(f"\n{'#' * 20} {group}  @ {week} {'#' * 20}")
    g, w = group.replace("'", "''"), week

    print("-- the event")
    q(f"""SELECT published_week, old_price, new_price, ROUND(pct_change, 1) AS pct, effective_date, publication_lag_days AS lag,
                 explanation_codes, mix_only, ndc_confirmed_share, n_ndcs_priced, n_distinct_prices, after_gap,
                 outcome_price_4w, outcome_price_8w, outcome_price_12w, outcome_reverted
          FROM price_events WHERE drug_group = '{g}' AND published_week = DATE'{w}'""")

    print("-- all events of this group (how jumpy is it?)")
    q(f"""SELECT published_week, old_price, new_price, ROUND(pct_change, 1) AS pct, explanation_codes, mix_only
          FROM price_events WHERE drug_group = '{g}' ORDER BY published_week""")

    print("-- group weekly series, 8 weeks before to 12 after (median price, NDC count)")
    q(f"""SELECT as_of_date, median_price, n_ndcs_priced, n_distinct_prices, price_changed
          FROM group_weekly WHERE drug_group = '{g}'
            AND as_of_date BETWEEN DATE_SUB(DATE'{w}', 56) AND DATE_ADD(DATE'{w}', 84) ORDER BY as_of_date""")

    print("-- every NDC of the group: own price at -4w, -1w, event week, +4w, +8w, +12w; codes and dates at the event")
    q(f"""WITH r AS (
              SELECT n.ndc, n.as_of_date, n.nadac_per_unit AS p, n.explanation_codes, n.effective_date,
                     n.generic_nadac_per_unit
              FROM nadac n JOIN ndc_dim d ON d.ndc = n.ndc WHERE d.drug_group = '{g}'),
          life AS (SELECT ndc, MIN(as_of_date) AS first_seen, MAX(as_of_date) AS last_seen FROM r GROUP BY ndc)
          SELECT r0.ndc, l.first_seen, l.last_seen,
                 MAX(CASE WHEN r.as_of_date = DATE_SUB(DATE'{w}', 28) THEN r.p END) AS p_m4w,
                 MAX(CASE WHEN r.as_of_date = DATE_SUB(DATE'{w}', 7)  THEN r.p END) AS p_m1w,
                 MAX(CASE WHEN r.as_of_date = DATE'{w}'                THEN r.p END) AS p_event,
                 MAX(CASE WHEN r.as_of_date = DATE_ADD(DATE'{w}', 28)  THEN r.p END) AS p_p4w,
                 MAX(CASE WHEN r.as_of_date = DATE_ADD(DATE'{w}', 56)  THEN r.p END) AS p_p8w,
                 MAX(CASE WHEN r.as_of_date = DATE_ADD(DATE'{w}', 84)  THEN r.p END) AS p_p12w,
                 MAX(CASE WHEN r.as_of_date = DATE'{w}' THEN CAST(r.explanation_codes AS STRING) END) AS codes_event,
                 MAX(CASE WHEN r.as_of_date = DATE'{w}' THEN r.effective_date END) AS effective_event,
                 MAX(CASE WHEN r.as_of_date = DATE'{w}' THEN r.generic_nadac_per_unit END) AS generic_price_event
          FROM (SELECT DISTINCT ndc FROM r) r0
          JOIN life l ON l.ndc = r0.ndc
          LEFT JOIN r ON r.ndc = r0.ndc
          GROUP BY r0.ndc, l.first_seen, l.last_seen ORDER BY r0.ndc""")

    print("-- description / unit / class / OTC ever used by these NDCs (all years)")
    q(f"""SELECT n.ndc, n.description, n.pricing_unit, n.classification, n.is_otc,
                 MIN(n.as_of_date) AS from_week, MAX(n.as_of_date) AS to_week, COUNT(*) AS weeks
          FROM nadac n JOIN ndc_dim d ON d.ndc = n.ndc WHERE d.drug_group = '{g}'
          GROUP BY 1, 2, 3, 4, 5 ORDER BY 1, from_week""")

    print("-- same description in OTHER groups (different unit / class) during the event week")
    q(f"""SELECT n.drug_group, COUNT(*) AS ndcs, MIN(n.nadac_per_unit) AS min_price, MAX(n.nadac_per_unit) AS max_price
          FROM nadac n
          WHERE n.as_of_date = DATE'{w}' AND n.description = SPLIT_PART('{g}', '|', 1) AND n.drug_group <> '{g}'
          GROUP BY 1""")

spark.stop()
