from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
import os

# === Setup Spark ===
spark = SparkSession.builder \
    .appName("IMM-Demo") \
    .config("spark.jars","jars/Spark-DIMM-assembly-1.0.jar")\
    .getOrCreate()

from scalablexplain.imm.dimm.pydimm import PyIMMWrapper  

# === Step 1: Load Dataset ===
# We'll use the Iris dataset from UCI
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
pdf = pd.DataFrame(iris.data, columns=iris.feature_names)
pdf["label"] = iris.target
df = spark.createDataFrame(pdf)

# === Step 2: Vectorize Features ===
feature_cols = iris.feature_names
vec_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vec = vec_assembler.transform(df).select("features")

# === Step 3: Run KMeans Clustering ===
kmeans = KMeans(k=3, seed=1, featuresCol="features", predictionCol="cluster")
model = kmeans.fit(df_vec)
centers = model.clusterCenters()
df_clustered = model.transform(df_vec).select("features", "cluster")

# === Step 4: Convert to IMM format ===
def to_instance(row):
    # You'll need a matching JVM-side case class: Instance(features: Vector, clusterId: Int, weight: Double)
    return (row["features"], int(row["cluster"]), 1.0)

clustered_rdd = df_clustered.rdd.map(to_instance)

# === Step 5: Run IMM Wrapper ===
jar_path = "jars/Spark-DIMM-assembly-1.0.jar"
imm = PyIMMWrapper(spark, jar_path)

# Cluster centers as list of arrays
center_vectors = [Vectors.dense(c.tolist()) for c in centers]

# Run IMM
tree = imm.run(clustered_rdd, center_vectors, num_splits=32, max_bins=32, seed=42)

# === Step 6: Print Tree Summary ===
print("\n===== IMM Tree Output =====\n")
for node_id, node in tree.items():
    print(f"Node {node_id}: {node}")

# === Optionally: Export to DOT file ===
# If your IMM Scala tree includes a toGraphviz method
if hasattr(imm.service, "exportTreeToDot"):  # if exposed
    dot = imm.service.exportTreeToDot(tree)
    with open("imm_tree.dot", "w") as f:
        f.write(dot)
    print("DOT file saved as imm_tree.dot")
