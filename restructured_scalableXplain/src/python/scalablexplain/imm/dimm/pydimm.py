from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from py4j.java_gateway import java_import

class PyIMMWrapper:
    def __init__(self, spark: SparkSession, jar_path: str):
        self.spark = spark
        self.sc = spark.sparkContext
        self.gateway = self.sc._gateway
        self.jvm = self.gateway.jvm

        # Load your custom jar
        self.sc.addPyFile(jar_path)

        # Import classes
        java_import(self.jvm, "dimm.api.IMMService")
        java_import(self.jvm, "dimm.core.Instance")

        self.service = self.jvm.dimm.api.IMMService()

    def run(self, clustered_rdd, cluster_centers, num_splits=32, max_bins=32, seed=42):
        java_centers = self.gateway.new_array(self.jvm.org.apache.spark.ml.linalg.Vector, len(cluster_centers))
        for i, vec in enumerate(cluster_centers):
            java_centers[i] = Vectors.dense(vec)

        tree = self.service.runIMMFromPython(clustered_rdd._jrdd, java_centers, num_splits, max_bins, seed)
        return tree
