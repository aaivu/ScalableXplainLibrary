from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from scalablexplain.imm.spark_cluster import DistributedIMMExplainer
import os
import urllib.request

# ---- Configuration ----
JAR_PATH = "jars/spark-dimm-assembly.jar"
IRIS_CSV = "iris.csv"
IRIS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
K = 5  # Number of clusters
DOT_OUTPUT = "imm_tree.dot"
FEATURE_COLS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# ---- Start Spark ----
spark = SparkSession.builder \
    .appName("Test DIMM on Iris") \
    .config("spark.jars", JAR_PATH) \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ---- Download Iris Dataset if Needed ----
if not os.path.exists(IRIS_CSV):
    print("Downloading Iris dataset...")
    urllib.request.urlretrieve(IRIS_URL, IRIS_CSV)

# ---- Load Data ----
iris_df = spark.read.csv(IRIS_CSV, header=True, inferSchema=True)

# ---- Vectorize Features ----
vec_assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
df = vec_assembler.transform(iris_df)

# ---- KMeans Clustering ----
kmeans = KMeans(k=K, seed=42, featuresCol="features", predictionCol="prediction")
model = kmeans.fit(df)
clustered_df = model.transform(df)

print("\n=== Cluster Centers ===")
for i, center in enumerate(model.clusterCenters()):
    print(f"Cluster {i}: {center}")

# ---- Run DIMM ----
explainer = DistributedIMMExplainer(model, num_splits=16, max_bins=16, seed=42)
tree, splits, tree_str = explainer.explain(clustered_df)

# ---- Output Tree ----
explainer.print_tree(tree_str)

# ---- Optional: Export .dot for Graphviz ----
explainer.export_graphviz(tree_str, DOT_OUTPUT)
print(f"\nDOT file exported to: {DOT_OUTPUT}")

# ---- Stop Spark ----
spark.stop()

from graphviz import Source

dot_path = "imm_tree.dot"         # your .dot file
output_path = "imm_tree"

# Render and open
src = Source.from_file(dot_path)
src.render(output_path, format="png", view=True)