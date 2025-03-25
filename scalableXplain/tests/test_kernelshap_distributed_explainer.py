import unittest
import sys
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import rand, broadcast, col, lit
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.ml.functions import vector_to_array

# Import the distributed explainer
# Adjust path if your code is structured differently:
from scalableXplain.explainers.distributed.distributed_kernel_shap import DistributedTabularSHAP
from scalableXplain.plots.bar_plot import BarPlot


# -------------------------
# 1) Define the UDF at module level
# -------------------------

def vector_access_udf(v, i):
    """Extract float(v[i]) from a Spark ML vector, safe for pickling."""
    return float(v[i])


vec_access = F.udf(vector_access_udf, FloatType())


class TestDistributedTabularSHAPIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Create Spark Session with SynapseML on the classpath
        cls.spark = (
            SparkSession.builder
            .appName("TestDistributedTabularSHAPIris")
            .master("local[*]")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.10")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .getOrCreate()
        )

        # 2. Load Iris as a Spark DataFrame
        iris_csv_path = "../datasets/iris.csv"  # Adjust path to your actual CSV
        df_spark = cls.spark.read.csv(iris_csv_path, header=True, inferSchema=True)

        print("Spark DF columns:", df_spark.columns)

        # For demonstration, let's see if we have "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm",
        # "Species" We'll treat "Species" as the label that we index
        label_indexer = StringIndexer(inputCol="Species", outputCol="label").fit(df_spark)

        # 4. Vector Assemble for logistic regression
        numeric_features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        assembler = VectorAssembler(
            inputCols=numeric_features,
            outputCol="features"
        )
        lr = LogisticRegression(featuresCol="features", labelCol="label")

        pipeline = Pipeline(stages=[label_indexer, assembler, lr])
        cls.model = pipeline.fit(df_spark)

        # Keep entire dataset for explanation, but pick a random subset for demonstration
        cls.full_df = df_spark
        cls.explain_instances = (
            cls.model.transform(df_spark)
            .orderBy(rand())
            .limit(5)
            .cache()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_tabular_shap_iris(self):
        """
                Use DistributedTabularSHAP on the Iris dataset
                and visualize the results for a small subset.
                """
        # 1. Create small random background dataset
        background_data = broadcast(
            self.full_df.orderBy(rand()).limit(100).cache()
        )

        # 2. Build the distributed TabularSHAP
        numeric_features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        shap_explainer = DistributedTabularSHAP(
            model=self.model,
            data=self.full_df,
            input_cols=numeric_features,
            target_col="probability",  # logistic regression outputs this vector
            target_classes=[1],  # let's explain class index=1
            num_samples=2000,
            background_data=background_data
        )

        # 3. Explain the subset
        shap_values = shap_explainer.explain(data=self.full_df)

        print("ll", shap_values)

        # ------




if __name__ == "__main__":
    unittest.main(argv=sys.argv, verbosity=2, exit=False)
