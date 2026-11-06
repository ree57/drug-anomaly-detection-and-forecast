"""Step 4: clean one year of NADAC data and save it as Parquet.

clean_nadac() only transforms and validates (no writing), so it can be reused for other years.
"""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession, functions as F

# Original CSV header -> short snake_case name. Everything is still text at this point.
RENAME = {
    "NDC Description": "description",
    "NDC": "ndc",
    "NADAC Per Unit": "nadac_per_unit",
    "Effective Date": "effective_date",
    "Pricing Unit": "pricing_unit",
    "Pharmacy Type Indicator": "pharmacy_type_indicator",
    "OTC": "otc",
    "Explanation Code": "explanation_code",
    "Classification for Rate Setting": "classification",
    "Corresponding Generic Drug NADAC Per Unit": "generic_nadac_per_unit",
    "Corresponding Generic Drug Effective Date": "generic_effective_date",
    "As of Date": "as_of_date",
}
PRICING_UNITS = ["EA", "ML", "GM"]
CLASSIFICATIONS = ["G", "B", "B-ANDA", "B-BIO"]
# Single M/d accept 1 or 2 digits: most dates are "12/18/2024", but 2022 has a week written "1/12/2022"
DATE_FORMAT = "M/d/yyyy"


def normalize_description(col):
    """Uppercase, trim, and collapse repeated spaces, so tiny text differences don't split a drug."""
    return F.regexp_replace(F.upper(F.trim(col)), r"\s+", " ")


def drug_group(description, pricing_unit, classification):
    """Key for 'the same drug product, priced together' (chosen in step 4, part 3)."""
    return F.concat_ws("|", normalize_description(description), pricing_unit, classification)


def _ok(check):
    """A check passes only when it is true. NULL (e.g. a value that didn't parse) counts as a fail."""
    return F.coalesce(check, F.lit(False))


def _date(name):
    # try_to_date returns NULL for bad text instead of crashing (Spark 4 runs in ANSI mode)
    return F.try_to_date(F.col(name), DATE_FORMAT)


def clean_nadac(spark, csv_path, year):
    """Return (clean_df, rejects_df) for one year's NADAC CSV."""
    raw = spark.read.csv(csv_path, header=True)  # no inferSchema: every column arrives as text
    # 2022-2023 files write some headers with underscores ("NADAC_Per_Unit"); match either spelling
    actual = {col.replace("_", " "): col for col in raw.columns}
    missing = set(RENAME) - set(actual)
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {sorted(missing)}")
    text = raw.select([F.trim(F.col(f"`{actual[old]}`")).alias(new) for old, new in RENAME.items()])

    # Pharmacy Type Indicator is always "C/I" (CMS: other types are "not used at this time"),
    # so it carries no information. Verify that instead of assuming it, then drop it.
    other_types = text.filter(~_ok(F.col("pharmacy_type_indicator") == "C/I")).count()
    if other_types:
        raise ValueError(f"{other_types} rows have a pharmacy type other than 'C/I'; review before dropping it")

    c = F.col
    parsed = text.select(
        "*",
        c("nadac_per_unit").try_cast("double").alias("_price"),
        _date("effective_date").alias("_effective"),
        _date("as_of_date").alias("_as_of"),
        c("generic_nadac_per_unit").try_cast("double").alias("_generic_price"),
        _date("generic_effective_date").alias("_generic_effective"),
    )

    # Each check is "what a good row looks like". A row failing any of them is rejected.
    checks = {
        "bad_ndc": _ok(c("ndc").rlike(r"^\d{11}$")),
        "missing_description": _ok(F.length("description") > 0),
        "bad_price": _ok(c("_price") > 0),
        "bad_effective_date": c("_effective").isNotNull(),
        "bad_as_of_date": c("_as_of").isNotNull(),
        "as_of_outside_year": c("_as_of").isNull() | _ok(F.year("_as_of") == year),
        "bad_pricing_unit": _ok(c("pricing_unit").isin(PRICING_UNITS)),
        "bad_otc": _ok(c("otc").isin("Y", "N")),
        "bad_classification": _ok(c("classification").isin(CLASSIFICATIONS)),
        "bad_explanation_code": c("explanation_code").isNull()
                                | _ok(c("explanation_code").rlike(r"^\d+(\s*,\s*\d+)*$")),
        "bad_generic_price": c("generic_nadac_per_unit").isNull() | _ok(c("_generic_price") > 0),
        "bad_generic_effective_date": c("generic_effective_date").isNull() | c("_generic_effective").isNotNull(),
    }
    # concat_ws skips NULLs, so this lists every failed check, e.g. "bad_ndc; bad_price"
    reason = F.concat_ws("; ", *[F.when(~passed, F.lit(name)) for name, passed in checks.items()])
    flagged = parsed.withColumn("reject_reason", reason)

    rejects = (flagged.filter(c("reject_reason") != "")
               .select(*RENAME.values(), "reject_reason", F.lit(year).alias("source_year")))

    clean = flagged.filter(c("reject_reason") == "").select(
        "description",
        "ndc",
        c("_price").alias("nadac_per_unit"),
        c("_effective").alias("effective_date"),
        "pricing_unit",
        (c("otc") == "Y").alias("is_otc"),
        "explanation_code",
        F.transform(F.split("explanation_code", ","),
                    lambda code: F.trim(code).try_cast("int")).alias("explanation_codes"),  # "1, 5" -> [1, 5]
        "classification",
        c("_generic_price").alias("generic_nadac_per_unit"),
        c("_generic_effective").alias("generic_effective_date"),
        c("_as_of").alias("as_of_date"),
        drug_group(c("description"), c("pricing_unit"), c("classification")).alias("drug_group"),
        F.lit(year).alias("source_year"),
    )

    # One row per NDC per week. A duplicate means we misunderstand the file, so stop, don't guess.
    dupes = clean.groupBy("ndc", "as_of_date").count().filter("count > 1")
    examples = dupes.limit(5).collect()
    if examples:
        raise ValueError(f"duplicate (ndc, as_of_date) pairs, e.g. {examples}")

    return clean, rejects


if __name__ == "__main__":
    # Usage: python clean.py 2022 2023 2024 2025 2026   (no arguments = 2025)
    YEARS = [int(y) for y in sys.argv[1:]] or [2025]
    CLEAN_DIR, REJECTS_DIR = "data/clean/nadac", "data/rejects/nadac"

    spark = (SparkSession.builder.master("local[*]").appName("clean")
             # rewriting one year replaces only its source_year=YYYY folder, never other years
             .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    for year in YEARS:
        csv_path = f"data/raw/nadac_{year}.csv"
        clean, rejects = clean_nadac(spark, csv_path, year)
        rejects = rejects.cache()
        clean.write.mode("overwrite").partitionBy("source_year").parquet(CLEAN_DIR)
        rejects.write.mode("overwrite").partitionBy("source_year").parquet(REJECTS_DIR)

        # ---- per-year quality report, read back from disk so it describes what was actually saved ----
        out = spark.read.parquet(CLEAN_DIR).filter(F.col("source_year") == year)
        rows_in, rows_out, n_rejects = spark.read.csv(csv_path, header=True).count(), out.count(), rejects.count()
        print(f"\n==== {year}: rows in {rows_in:,} | rows out {rows_out:,} | rejects {n_rejects:,} | "
              f"all accounted for: {rows_in == rows_out + n_rejects}")
        if n_rejects:
            rejects.groupBy("reject_reason").count().orderBy(F.desc("count")).show(truncate=False)
        nulls = out.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in out.columns]).first().asDict()
        print("  columns with NULLs:", {c: n for c, n in nulls.items() if n} or "none")
        rejects.unpersist()

    print("\npartition folders:", sorted(os.listdir(CLEAN_DIR)))
    spark.stop()
