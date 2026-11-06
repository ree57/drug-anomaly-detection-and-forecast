import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable  # make Spark's workers use this venv's Python
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("smoke_test").getOrCreate()

rows = [
    ("Atorvastatin 20mg", 0.04), ("Metformin 500mg", 0.02), ("Lisinopril 10mg", 0.03),
    ("Amoxicillin 500mg", 0.09), ("Insulin Glargine", 26.50),
]
df = spark.createDataFrame(rows, ["drug_name", "price_per_unit"])
df.show()

df.write.mode("overwrite").parquet("smoke_output.parquet")
spark.read.parquet("smoke_output.parquet").show()

spark.stop()
