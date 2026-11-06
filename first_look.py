import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("first_look").getOrCreate()

df = spark.read.csv("data/raw/nadac_2025.csv", header=True, inferSchema=True)

df.printSchema()
print("rows:", df.count())
df.show(10, truncate=False)

spark.stop()
