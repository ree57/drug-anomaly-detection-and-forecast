"""Step 4, part 3: test candidate drug-group keys.

Question: within a group, in a given week, do all NDCs share one identical NADAC?
"""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession, functions as F
from clean import clean_nadac, normalize_description

spark = SparkSession.builder.master("local[*]").appName("group_check").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

clean, _ = clean_nadac(spark, "data/raw/nadac_2025.csv", 2025)
df = (clean.withColumn("desc_norm", normalize_description(F.col("description")))
           .withColumn("has_code_5", F.array_contains("explanation_codes", 5))
           .cache())

CANDIDATES = {
    "description": ["desc_norm"],
    "description + unit": ["desc_norm", "pricing_unit"],
    "description + unit + class": ["desc_norm", "pricing_unit", "classification"],
}


def group_weeks(keys):
    """One row per (group, week): how many NDCs, how many distinct prices."""
    return df.groupBy(*keys, "as_of_date").agg(
        F.countDistinct("ndc").alias("ndcs"),
        F.countDistinct("nadac_per_unit").alias("prices"),
        F.count("*").alias("rows"),
        (F.max("nadac_per_unit") / F.min("nadac_per_unit")).alias("max_over_min"))


print("\n==== how often does a group share ONE price in a week? ====")
for name, keys in CANDIDATES.items():
    gw = group_weeks(keys).cache()
    multi = gw.filter("ndcs > 1")  # groups with a single NDC pass trivially, so judge on multi-NDC ones
    r = multi.agg(
        F.count("*").alias("group_weeks"),
        F.avg((F.col("prices") == 1).cast("int")).alias("share"),
        (F.sum(F.when(F.col("prices") == 1, F.col("rows"))) / F.sum("rows")).alias("row_share"),
    ).first()
    print(f"{name:28s} groups={df.select(*keys).distinct().count():6,d}  "
          f"multi-NDC group-weeks={r['group_weeks']:7,d}  one price: {r['share']:.1%} of group-weeks, "
          f"{r['row_share']:.1%} of rows")

keys = CANDIDATES["description + unit + class"]
gw = group_weeks(keys).cache()
failing = gw.filter("prices > 1")

print("\n==== chosen candidate: how big are the price differences where it fails? ====")
failing.select(F.percentile_approx("max_over_min", [0.5, 0.9, 0.99]).alias("max/min price: p50, p90, p99")).show(truncate=False)

print("Do failing groups involve explanation code 5 (priced by package size)?")
(df.join(failing.select(*keys, "as_of_date"), keys + ["as_of_date"], "left_semi")
   .agg(F.avg(F.col("has_code_5").cast("int")).alias("share_of_rows_with_code_5_in_failing_groups"))
   .show())
df.agg(F.avg(F.col("has_code_5").cast("int")).alias("share_of_rows_with_code_5_overall")).show()

print("Groups that fail most often:")
worst = (failing.groupBy(*keys).agg(F.count("*").alias("failing_weeks"), F.max("prices").alias("max_prices"),
                                    F.max("max_over_min").alias("max_over_min"))
                .orderBy(F.desc("failing_weeks"), F.desc("max_over_min")))
worst.show(8, truncate=False)

print("Same key, ignoring rows priced by package size (code 5):")
no5 = (df.filter(~F.col("has_code_5")).groupBy(*keys, "as_of_date")
         .agg(F.countDistinct("ndc").alias("ndcs"), F.countDistinct("nadac_per_unit").alias("prices"))
         .filter("ndcs > 1"))
no5.agg(F.count("*").alias("group_weeks"), F.avg((F.col("prices") == 1).cast("int")).alias("share_one_price")).show()

# What is going on inside each failing group-week? (mixed OTC/Rx, any package-size pricing)
why = (df.join(failing.select(*keys, "as_of_date"), keys + ["as_of_date"], "left_semi")
         .groupBy(*keys, "as_of_date")
         .agg(F.countDistinct("is_otc").alias("otc_kinds"), F.max(F.col("has_code_5").cast("int")).alias("any_code_5")))
by_group = why.groupBy(*keys).agg(F.max("otc_kinds").alias("otc_kinds"), F.max("any_code_5").alias("any_code_5"))
ranked = worst.join(by_group, keys)
picks = [ranked.filter("any_code_5 = 1").orderBy(F.desc("failing_weeks"), F.desc("max_over_min")).first(),  # package size
         ranked.filter("otc_kinds > 1").orderBy(F.desc("failing_weeks")).first(),                            # OTC + Rx mixed
         ranked.filter("any_code_5 = 0 and otc_kinds = 1").orderBy(F.desc("failing_weeks")).first()]         # neither

print("Three examples, one failing week each:")
for ex in [p for p in picks if p is not None]:
    week = failing.filter((F.col("desc_norm") == ex["desc_norm"]) & (F.col("pricing_unit") == ex["pricing_unit"])
                          & (F.col("classification") == ex["classification"])).agg(F.max("as_of_date")).first()[0]
    print(f"--- {ex['desc_norm']} | {ex['pricing_unit']} | {ex['classification']}  (week {week})")
    # one row per price tier inside the group that week
    (df.filter((F.col("desc_norm") == ex["desc_norm"]) & (F.col("pricing_unit") == ex["pricing_unit"])
               & (F.col("classification") == ex["classification"]) & (F.col("as_of_date") == week))
       .groupBy("nadac_per_unit")
       .agg(F.count("*").alias("ndcs"),
            F.collect_set("is_otc").alias("is_otc"),
            F.collect_set("explanation_code").alias("explanation_codes"),
            F.slice(F.sort_array(F.collect_list("ndc")), 1, 4).alias("example_ndcs"))
       .orderBy("nadac_per_unit").show(truncate=False))

spark.stop()
