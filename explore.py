"""Step 3: explore how NADAC prices behave in 2025. Exploration only, not a pipeline."""
import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession, Window, functions as F

spark = SparkSession.builder.master("local[*]").appName("explore").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def to_date(col):
    # the file stores dates as text like "12/18/2024"; this pattern tells Spark how to read them
    return F.to_date(F.col(col), "MM/dd/yyyy")


# ---- 1. Typed read: no inferSchema, so every column comes in as text (NDC keeps its zeros) ----
raw = spark.read.csv("data/raw/nadac_2025.csv", header=True)
df = raw.select(
    F.col("NDC").alias("ndc"),
    F.col("NDC Description").alias("desc"),
    F.col("NADAC Per Unit").cast("double").alias("price"),
    to_date("Effective Date").alias("eff_date"),
    F.col("Pricing Unit").alias("unit"),
    F.col("Explanation Code").alias("expl"),
    F.col("Classification for Rate Setting").alias("cls"),
    to_date("As of Date").alias("as_of"),
).cache()

header("1. schema + sanity checks")
df.printSchema()
df.select(
    F.count("*").alias("rows"),
    F.sum(F.col("eff_date").isNull().cast("int")).alias("unparsed_eff_dates"),
    F.sum(F.col("as_of").isNull().cast("int")).alias("unparsed_as_of"),
    F.sum(F.col("price").isNull().cast("int")).alias("null_prices"),
    F.min("as_of").alias("first_week"), F.max("as_of").alias("last_week"),
).show()
dupes = df.groupBy("ndc", "as_of").count().filter("count > 1").count()
print("NDC+week pairs appearing more than once:", dupes)

# ---- Change detection: compare each week's row with the same NDC's previous week ----
w = Window.partitionBy("ndc").orderBy("as_of")
weekly = (df
          .withColumn("prev_price", F.lag("price").over(w))
          .withColumn("prev_eff", F.lag("eff_date").over(w))
          .withColumn("prev_unit", F.lag("unit").over(w)))
changes = (weekly
           .filter(F.col("prev_price").isNotNull() & (F.col("price") != F.col("prev_price")))
           .withColumn("pct", F.round((F.col("price") / F.col("prev_price") - 1) * 100, 2))
           .cache())

# one row per NDC: latest classification, number of effective dates, number of real price changes
per_ndc = (df.groupBy("ndc").agg(
               F.max_by("desc", "as_of").alias("desc"),
               F.max_by("cls", "as_of").alias("cls"),
               F.countDistinct("eff_date").alias("n_eff_dates"),
               F.min("price").alias("min_price"), F.max("price").alias("max_price"),
               F.max_by("price", "as_of").alias("last_price"))
           .join(changes.groupBy("ndc").agg(F.count("*").alias("n_changes")), "ndc", "left")
           .fillna(0, ["n_changes"])
           .cache())


def bucket(col):
    return (F.when(F.col(col) == 0, "0").when(F.col(col) == 1, "1")
             .when(F.col(col) == 2, "2").otherwise("3+"))


header("a. how often does an NDC's price change in 2025?")
print("Real price changes per NDC (price differs from previous week):")
per_ndc.groupBy(bucket("n_changes").alias("changes")).count().orderBy("changes").show()
print("Distinct Effective Dates per NDC (what you asked for literally):")
per_ndc.groupBy(bucket("n_eff_dates").alias("eff_dates")).count().orderBy("eff_dates").show()
print("Weeks where Effective Date moved but the price stayed the same:",
      weekly.filter((F.col("eff_date") != F.col("prev_eff")) & (F.col("price") == F.col("prev_price"))).count())

header("b. size of a price change, in %")
print("Total price changes:", changes.count())
qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
pct = changes.select(F.percentile_approx("pct", qs, 100000)).first()[0]
print(dict(zip(["p1", "p5", "p25", "median", "p75", "p95", "p99"], pct)))
changes.select(
    F.round(F.avg((F.col("pct") > 0).cast("int")) * 100, 1).alias("pct_of_changes_that_are_increases"),
    F.round(F.avg((F.abs("pct") < 5).cast("int")) * 100, 1).alias("pct_of_changes_under_5pct"),
).show()

header("c. brand vs generic")
freq = per_ndc.groupBy("cls").agg(
    F.count("*").alias("ndcs"),
    F.round(F.avg("n_changes"), 2).alias("avg_changes_per_ndc"),
    F.round(F.avg((F.col("n_changes") > 0).cast("int")) * 100, 1).alias("pct_ndcs_that_changed"))
size = changes.groupBy("cls").agg(
    F.count("*").alias("changes"),
    F.percentile_approx(F.abs("pct"), 0.5).alias("median_abs_pct"),
    F.percentile_approx(F.abs("pct"), 0.95).alias("p95_abs_pct"),
    F.round(F.avg((F.col("pct") > 0).cast("int")) * 100, 1).alias("pct_increases"))
freq.join(size, "cls", "left").orderBy(F.desc("ndcs")).show()

header("d. biggest single jumps and drops")
cols = ["desc", "ndc", "prev_unit", "unit", "prev_price", "price", "pct",
        "prev_eff", "eff_date", "expl", "min_price", "max_price", "last_price"]
top = changes.join(per_ndc.select("ndc", "min_price", "max_price", "last_price"), "ndc")
print("Top 10 jumps (price went UP the most, %):")
top.orderBy(F.desc("pct")).select(cols).show(10, truncate=False)
print("Top 10 drops (price went DOWN the most, %):")
top.orderBy("pct").select(cols).show(10, truncate=False)

header("e. Repatha, Otezla, Prolia")
brands = df.filter(F.upper("desc").rlike("REPATHA|OTEZLA|PROLIA"))
print("Rows found:", brands.count())
(brands.groupBy("desc", "ndc", "unit", "cls", "eff_date", "price")
       .agg(F.min("as_of").alias("first_week"), F.max("as_of").alias("last_week"),
            F.count("*").alias("weeks"))
       .orderBy("desc", "ndc", "eff_date")
       .show(60, truncate=False))

header("f. the most expensive unit")
top_ndc = df.orderBy(F.desc("price")).select("ndc").first()[0]
(df.filter(F.col("ndc") == top_ndc)
   .groupBy("desc", "ndc", "unit", "cls", "expl", "eff_date", "price")
   .agg(F.min("as_of").alias("first_week"), F.max("as_of").alias("last_week"), F.count("*").alias("weeks"))
   .orderBy("eff_date").show(truncate=False))
print("Other NDCs priced over $5,000 per unit:")
(per_ndc.filter("max_price > 5000").select("desc", "ndc", "cls", "max_price")
        .join(df.select("ndc", "unit").distinct(), "ndc")
        .orderBy(F.desc("max_price")).show(15, truncate=False))

spark.stop()
