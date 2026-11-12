"""Step 6a: build analysis tables with Spark SQL from the cleaned NADAC data.

Each table's query lives in sql/<name>.sql; results go to data/tables/<name>/ as Parquet.
"""
import os
import sys
import time
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

TABLES = ["ndc_dim", "group_dim", "group_weekly", "price_events"]  # dependency order: later read earlier

spark = (SparkSession.builder.master("local[*]").appName("build_tables")
         .config("spark.driver.memory", "4g")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# make the clean data queryable from SQL under the name "nadac" (all years at once)
spark.read.parquet("data/clean/nadac").createOrReplaceTempView("nadac")

for name in TABLES:
    start = time.time()
    query = Path(f"sql/{name}.sql").read_text(encoding="utf-8")
    spark.sql(query).write.mode("overwrite").parquet(f"data/tables/{name}")
    # register the SAVED table under its name, so the next SQL file reads it from disk, not recomputes it
    spark.read.parquet(f"data/tables/{name}").createOrReplaceTempView(name)
    print(f"{name}: {spark.table(name).count():,} rows ({time.time() - start:.0f}s)")

spark.stop()
