# test.py
import os
from pyspark.sql import SparkSession

# Define the full path to the local jars directory
jars_dir = os.path.join(os.getcwd(), "jars")

# Collect all jar files in that directory
jar_files = [os.path.join(jars_dir, f) for f in os.listdir(jars_dir) if f.endswith(".jar")]

# Join them into a comma-separated string
jars_str = ",".join(jar_files)

print(f"JARS STRING: {jars_str} ")
# Initialize Spark with the local JARs
spark = SparkSession.builder \
    .appName("Test") \
    .config("spark.jars", jars_str) \
    .getOrCreate()

# Test DataFrame
df = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
df.show()
