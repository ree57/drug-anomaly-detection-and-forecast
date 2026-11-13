"""Step 10c: past-only competition features per (drug group, origin t), from the NDC labeler code (first 5 digits).

  labelers(g, w) = set of labeler codes among the group's priced NDCs in published week w
  at origin t, with "as of t-k" = the group's last priced week at or before t-k (same rule as Step 9):
    n_labelers          |labelers at t|
    d_labelers_12w      n_labelers - |labelers as of t-12|
    d_labelers_26w      n_labelers - |labelers as of t-26|
    labeler_exit_26w    labelers present as of t-26 but not at t
    labeler_entry_26w   labelers present at t but not as of t-26
    single_source       n_labelers == 1
Note: a labeler is whoever labels the package (manufacturer, repackager or store brand), not always the maker.

Leakage evidence: provenance (latest week read <= origin) and a truncation test on a sample, recomputed in
pandas from NADAC rows with every week after t deleted. Output: data/tables/forecast_competition/.
"""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
import pandas as pd
from pyspark.sql import SparkSession

SAMPLE_ROWS = 300
SEED = 42

spark = (SparkSession.builder.master("local[*]").appName("competition_features")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
spark.read.parquet("data/clean/nadac").createOrReplaceTempView("nadac")
spark.read.parquet("data/tables/ndc_dim").createOrReplaceTempView("ndc_dim")
spark.read.parquet("data/tables/forecast_dataset").createOrReplaceTempView("fd")

spark.sql("""
    WITH weeks AS (
        SELECT as_of_date, CAST(ROW_NUMBER() OVER (ORDER BY as_of_date) - 1 AS INT) AS wk
        FROM (SELECT DISTINCT as_of_date FROM nadac)),
    lab AS (   -- canonical group (as in group_weekly), one row per group per week
        SELECT d.drug_group, w.wk, ARRAY_SORT(COLLECT_SET(SUBSTR(n.ndc, 1, 5))) AS labelers
        FROM nadac n JOIN ndc_dim d ON d.ndc = n.ndc JOIN weeks w ON w.as_of_date = n.as_of_date
        GROUP BY d.drug_group, w.wk),
    asof AS (
        SELECT *,
            LAST(labelers, TRUE) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 12 PRECEDING) AS lab_12,
            LAST(wk, TRUE)       OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 12 PRECEDING) AS src_12,
            LAST(labelers, TRUE) OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 26 PRECEDING) AS lab_26,
            LAST(wk, TRUE)       OVER (PARTITION BY drug_group ORDER BY wk RANGE BETWEEN UNBOUNDED PRECEDING AND 26 PRECEDING) AS src_26
        FROM lab)
    SELECT f.drug_group, f.origin_week, f.origin_wk,
           SIZE(a.labelers)                                           AS n_labelers,
           SIZE(a.labelers) - SIZE(a.lab_12)                          AS d_labelers_12w,   -- NULL if no week <= t-12
           SIZE(a.labelers) - SIZE(a.lab_26)                          AS d_labelers_26w,
           SIZE(ARRAY_EXCEPT(a.lab_26, a.labelers))                   AS labeler_exit_26w,
           SIZE(ARRAY_EXCEPT(a.labelers, a.lab_26))                   AS labeler_entry_26w,
           CAST(SIZE(a.labelers) = 1 AS INT)                          AS single_source,
           GREATEST(a.wk, COALESCE(a.src_12, -1), COALESCE(a.src_26, -1)) AS comp_max_src_wk
    FROM fd f JOIN asof a ON a.drug_group = f.drug_group AND a.wk = f.origin_wk
""").write.mode("overwrite").parquet("data/tables/forecast_competition")

spark.read.parquet("data/tables/forecast_competition").createOrReplaceTempView("comp")
print("\n======== forecast_competition ========")
spark.sql("""SELECT COUNT(*) AS rows, (SELECT COUNT(*) FROM fd) AS forecast_rows,
                    MAX(comp_max_src_wk - origin_wk) AS max_src_minus_origin,
                    SUM(CASE WHEN comp_max_src_wk > origin_wk THEN 1 ELSE 0 END) AS provenance_violations,
                    SUM(CASE WHEN labeler_exit_26w IS NULL THEN 1 ELSE 0 END) AS rows_without_t_minus_26
             FROM comp""").show(truncate=False)
spark.sql("""SELECT CASE WHEN n_labelers >= 4 THEN '4+' ELSE CAST(n_labelers AS STRING) END AS n_labelers,
                    COUNT(*) AS rows FROM comp GROUP BY 1 ORDER BY 1""").show()

# truncation test: recompute from NADAC rows with every week after t deleted
sample = spark.table("comp").sample(fraction=0.003, seed=SEED).limit(SAMPLE_ROWS).toPandas()
groups = sample.drug_group.unique().tolist()
spark.createDataFrame(pd.DataFrame({"drug_group": groups})).createOrReplaceTempView("sg")
raw = spark.sql("""SELECT d.drug_group, n.as_of_date, n.ndc FROM nadac n JOIN ndc_dim d ON d.ndc = n.ndc
                   JOIN sg ON sg.drug_group = d.drug_group""").toPandas()
all_weeks = sorted(spark.sql("SELECT DISTINCT as_of_date FROM nadac").toPandas().as_of_date)
spark.stop()

wk_of = {w: i for i, w in enumerate(all_weeks)}
raw["wk"] = raw.as_of_date.map(wk_of)
raw["lab"] = raw.ndc.str[:5]
sets = raw.groupby(["drug_group", "wk"]).lab.agg(set)


def asof(g, wk):
    s = sets.loc[g]
    s = s[s.index <= wk]
    return s.iloc[-1] if len(s) else None


bad = 0
for _, r in sample.iterrows():
    now, l12, l26 = asof(r.drug_group, r.origin_wk), asof(r.drug_group, r.origin_wk - 12), asof(r.drug_group, r.origin_wk - 26)
    exp = {"n_labelers": len(now),
           "d_labelers_12w": None if l12 is None else len(now) - len(l12),
           "d_labelers_26w": None if l26 is None else len(now) - len(l26),
           "labeler_exit_26w": None if l26 is None else len(l26 - now),
           "labeler_entry_26w": None if l26 is None else len(now - l26),
           "single_source": int(len(now) == 1)}
    for c, v in exp.items():
        a = r[c]
        bad += not ((v is None and pd.isna(a)) or (v is not None and not pd.isna(a) and int(a) == v))
print(f"truncation test: {len(sample)} rows x 6 features compared, mismatches: {bad}")
