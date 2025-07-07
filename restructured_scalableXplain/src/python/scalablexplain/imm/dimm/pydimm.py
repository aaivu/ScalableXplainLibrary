from pyspark.sql import SparkSession
from pyspark import SparkContext
import numpy as np

class PyIMM:
    def __init__(self, spark, jar=None):
        if jar:
            spark.sparkContext.addPyFile(jar)
            spark.sparkContext._jsc.addJar(jar)
        self._spark   = spark
        self._jvm     = spark._jvm
        self._wrapper = self._jvm.dimm.wrapper.IMMWrapper

    @staticmethod
    def _to_vector(jvm, arr):
        """
        Convert a 1-D Python/NumPy array to
        org.apache.spark.ml.linalg.DenseVector (JVM).
        Works on every Spark / Scala version.
        """
        gw = SparkContext._gateway                    # active Py4J gateway
        dbl_arr = gw.new_array(gw.jvm.double, len(arr))
        for i, v in enumerate(arr):
            dbl_arr[i] = float(v)                     # primitive double value
        return jvm.org.apache.spark.ml.linalg.Vectors.dense(dbl_arr)

    def run(self, clustered_df, centers, num_splits=32, max_bins=32, seed=42):
        # Convert Python list/np.array of centres → java.util.List[Vector]
        j_centers = self._jvm.java.util.ArrayList()
        for c in centers:
            j_centers.add(self._to_vector(self._jvm, c))

        j_res = self._wrapper.runIMMFromClusteredDF(
            clustered_df._jdf,    # pass the JVM DataFrame
            j_centers,
            int(num_splits),
            int(max_bins),
            int(seed)
        )

        # back to Python
        tree   = {int(k): v for k, v in j_res.get("tree").entrySet()}
        splits = [[s for s in feat] for feat in j_res.get("splits")]
        return {"tree": tree, "splits": splits}
