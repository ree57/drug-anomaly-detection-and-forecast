"""Step 5: checks across all cleaned NADAC years in data/clean/nadac/."""
import os
import sys
from datetime import timedelta

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.master("local[*]").appName("cross_year_check")
         .config("spark.driver.memory", "4g")   # default 1g is too small for 7M rows (out-of-memory error)
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
df = spark.read.parquet("data/clean/nadac")   # all source_year=YYYY folders at once; Parquet rereads fast


def header(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


header("per year")
(df.groupBy("source_year").agg(
    F.count("*").alias("rows"),
    F.countDistinct("ndc").alias("ndcs"),
    F.countDistinct("drug_group").alias("groups"),
    F.countDistinct(F.when(F.col("classification") == "G", F.col("ndc"))).alias("generic_ndcs"),
    F.countDistinct(F.when(F.col("classification") != "G", F.col("ndc"))).alias("brand_ndcs"),
    F.countDistinct("as_of_date").alias("weeks"),
    F.min("as_of_date").alias("first_week"), F.max("as_of_date").alias("last_week"))
 .orderBy("source_year").show())

header("weekly as_of dates")
weeks = df.groupBy("as_of_date").agg(F.collect_set("source_year").alias("in_files"),
                                     F.dayofweek("as_of_date").alias("dow")).orderBy("as_of_date").collect()
dates = [w["as_of_date"] for w in weeks]
print("weeks found:", len(dates), "| first:", dates[0], "| last:", dates[-1])
print("not a Wednesday:", [str(w["as_of_date"]) for w in weeks if w["dow"] != 4] or "none")   # Spark: Sunday=1
gaps = [(str(a), str(b), (b - a).days) for a, b in zip(dates, dates[1:]) if (b - a).days != 7]
print("gaps between consecutive weeks that are not 7 days:", gaps or "none")
expected, d = set(), dates[0]
while d <= dates[-1]:
    expected.add(d)
    d += timedelta(days=7)
print("missing Wednesdays:", sorted(str(x) for x in expected - set(dates)) or "none")
print("weeks found in more than one yearly file:",
      [(str(w["as_of_date"]), sorted(w["in_files"])) for w in weeks if len(w["in_files"]) > 1] or "none")

header("(ndc, as_of_date) unique across ALL years?")
dupes = df.groupBy("ndc", "as_of_date").count().filter("count > 1")
print("duplicate pairs:", dupes.count())

header("drug_group continuity across years")
# each NDC's group in each year (its latest row that year)
per_year = df.groupBy("ndc", "source_year").agg(
    F.max_by("drug_group", "as_of_date").alias("drug_group"),
    F.max_by("description", "as_of_date").alias("description"),
    F.max_by("pricing_unit", "as_of_date").alias("unit"),
    F.max_by("classification", "as_of_date").alias("cls"))
multi_year = per_year.groupBy("ndc").agg(
    F.countDistinct("source_year").alias("years"),
    F.countDistinct("drug_group").alias("groups"),
    F.countDistinct("description").alias("descriptions"),
    F.countDistinct("unit").alias("units"),
    F.countDistinct("cls").alias("classes")).filter("years > 1").cache()
changed = multi_year.filter("groups > 1")
n_multi, n_changed = multi_year.count(), changed.count()
print(f"NDCs present in 2+ years: {n_multi:,} | of those, changed drug_group: {n_changed:,} ({n_changed / n_multi:.1%})")
print("what changed (an NDC can change more than one part):")
changed.select(
    F.sum((F.col("descriptions") > 1).cast("int")).alias("description_changed"),
    F.sum((F.col("units") > 1).cast("int")).alias("unit_changed"),
    F.sum((F.col("classes") > 1).cast("int")).alias("classification_changed")).show()

print("5 examples of a reworded description:")
reworded = changed.filter("descriptions > 1").select("ndc").limit(5)
(per_year.join(reworded, "ndc").orderBy("ndc", "source_year")
         .select("ndc", "source_year", "description", "unit", "cls").show(30, truncate=False))
print("5 examples of a classification change:")
reclassed = changed.filter("classes > 1 and descriptions = 1").select("ndc").limit(5)
(per_year.join(reclassed, "ndc").orderBy("ndc", "source_year")
         .select("ndc", "source_year", "description", "unit", "cls").show(30, truncate=False))


def history(title, ndc_filter):
    """One row per price segment (each distinct effective date + price), across all years."""
    header(title)
    (df.filter(ndc_filter)
       .groupBy("ndc", "description", "effective_date", "nadac_per_unit")
       .agg(F.min("as_of_date").alias("first_week"), F.max("as_of_date").alias("last_week"),
            F.count("*").alias("weeks"), F.collect_set("source_year").alias("files"))
       .orderBy("effective_date", "first_week")
       .show(100, truncate=False))


history("spot check: OTEZLA 30 MG TABLET (NDC 55513013760)", F.col("ndc") == "55513013760")

# generic: the atorvastatin 20 mg NDC that appears in the most weeks
atorva = (df.filter(F.col("description").rlike("^ATORVASTATIN.* 20 MG"))
            .groupBy("ndc", "description").agg(F.count("*").alias("weeks"))
            .orderBy(F.desc("weeks"), "ndc").first())
history(f"spot check: {atorva['description']} (NDC {atorva['ndc']}, {atorva['weeks']} weeks)",
        F.col("ndc") == atorva["ndc"])

spark.stop()
