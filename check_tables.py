"""Step 6a: checks on the tables built by build_tables.py."""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

spark = (SparkSession.builder.master("local[*]").appName("check_tables")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.read.parquet("data/clean/nadac").createOrReplaceTempView("nadac")
for name in ["ndc_dim", "group_dim", "group_weekly"]:
    spark.read.parquet(f"data/tables/{name}").createOrReplaceTempView(name)


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def q(sql, n=40):
    spark.sql(sql).show(n, truncate=False)


header("row counts")
q("""SELECT 'nadac (clean)' AS t, COUNT(*) AS rows FROM nadac
     UNION ALL SELECT 'ndc_dim', COUNT(*) FROM ndc_dim
     UNION ALL SELECT 'group_dim', COUNT(*) FROM group_dim
     UNION ALL SELECT 'group_weekly', COUNT(*) FROM group_weekly""")

header("every NDC maps to exactly one group")
q("""SELECT
       (SELECT COUNT(*) FROM ndc_dim)                                    AS ndc_dim_rows,
       (SELECT COUNT(DISTINCT ndc) FROM ndc_dim)                         AS distinct_ndcs_in_dim,
       (SELECT COUNT(DISTINCT ndc) FROM nadac)                           AS distinct_ndcs_in_clean,
       (SELECT COUNT(*) FROM ndc_dim WHERE drug_group IS NULL)           AS ndcs_without_group,
       (SELECT COUNT(*) FROM nadac n LEFT ANTI JOIN ndc_dim d ON n.ndc = d.ndc) AS clean_rows_without_ndc,
       (SELECT SUM(n_ndcs) FROM group_dim)                               AS sum_of_group_n_ndcs,
       (SELECT SUM(n_ndcs_priced) FROM group_weekly)                     AS rows_counted_in_group_weekly""")

header("group_weekly sanity")
q("""SELECT
       COUNT(*) - COUNT(DISTINCT drug_group, as_of_date)       AS duplicate_group_weeks,
       ROUND(AVG(CAST(price_changed AS INT)) * 100, 1)         AS pct_weeks_price_changed,
       SUM(CASE WHEN prev_price IS NULL THEN 1 ELSE 0 END)     AS first_weeks,
       SUM(CASE WHEN weeks_since_prev > 1 THEN 1 ELSE 0 END)   AS rows_after_a_gap,
       SUM(CASE WHEN n_distinct_prices > 1 THEN 1 ELSE 0 END)  AS rows_with_price_tiers
     FROM group_weekly""")
print("Gap lengths (weeks_since_prev):")
q("""SELECT weeks_since_prev, COUNT(*) AS rows FROM group_weekly
     WHERE weeks_since_prev IS NOT NULL GROUP BY weeks_since_prev ORDER BY weeks_since_prev""", 12)

header("reworded NDCs")
q("""SELECT COUNT(*) AS ndcs_with_more_than_one_raw_group,
            SUM(CASE WHEN groups_seen > 1 AND class_changed THEN 1 ELSE 0 END) AS of_which_class_changed
     FROM ndc_dim WHERE groups_seen > 1""")
EXAMPLE = "62756001740"  # TESTOSTERON CYP -> TESTOSTERONE CYP in 2023
print(f"BEFORE (raw drug_group in the clean data) for NDC {EXAMPLE}:")
q(f"""SELECT drug_group, MIN(as_of_date) AS first_week, MAX(as_of_date) AS last_week, COUNT(*) AS weeks
      FROM nadac WHERE ndc = '{EXAMPLE}' GROUP BY drug_group ORDER BY first_week""")
print("AFTER (ndc_dim) - one canonical group:")
q(f"SELECT ndc, drug_group, first_seen, last_seen, weeks_present, groups_seen FROM ndc_dim WHERE ndc = '{EXAMPLE}'")
print("AFTER (group_weekly) - that group's series is one continuous history:")
q(f"""SELECT g.drug_group, MIN(g.as_of_date) AS first_week, MAX(g.as_of_date) AS last_week, COUNT(*) AS weeks,
             MAX(g.weeks_since_prev) AS longest_gap_weeks
      FROM group_weekly g JOIN ndc_dim d ON g.drug_group = d.drug_group
      WHERE d.ndc = '{EXAMPLE}' GROUP BY g.drug_group""")
print("Does the old misspelled group still exist as its own group?")
q("SELECT COUNT(*) AS rows_in_group_dim FROM group_dim WHERE drug_group LIKE 'TESTOSTERON CYP %'")


def spot_check(label, ndc):
    header(f"spot check: {label}")
    print("Weeks where the group's median equals this NDC's own price:")
    q(f"""SELECT COUNT(*) AS weeks, SUM(CASE WHEN g.median_price = n.nadac_per_unit THEN 1 ELSE 0 END) AS same_price
          FROM nadac n JOIN ndc_dim d ON n.ndc = d.ndc
          JOIN group_weekly g ON g.drug_group = d.drug_group AND g.as_of_date = n.as_of_date
          WHERE n.ndc = '{ndc}'""")
    print("group_weekly rows where the price changed (plus the first week):")
    q(f"""SELECT g.as_of_date, g.prev_price, g.median_price, g.pct_change, g.n_ndcs_priced,
                 g.n_distinct_prices, g.weeks_since_prev
          FROM group_weekly g JOIN ndc_dim d ON g.drug_group = d.drug_group
          WHERE d.ndc = '{ndc}' AND (g.price_changed OR g.prev_price IS NULL)
          ORDER BY g.as_of_date""", 80)


spot_check("OTEZLA 30 MG TABLET", "55513013760")
spot_check("ATORVASTATIN 20 MG TABLET", "00093505998")
spark.stop()
