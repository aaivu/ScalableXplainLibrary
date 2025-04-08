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

# If your new LIME explainer is in a file named distributed_tabular_lime.py:
# Adjust path if your code is structured differently:
from scalableXplain.explainers.distributed.distributed_lime import DistributedTabularLIME


def vector_access_udf(v, i):
    """Extract float(v[i]) from a Spark ML vector, safe for pickling."""
    return float(v[i])


vec_access = F.udf(vector_access_udf, FloatType())


class TestDistributedTabularLIMEIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Create Spark Session with SynapseML on the classpath
        cls.spark = (
            SparkSession.builder
            .appName("TestDistributedTabularLIMEIris")
            .master("local[*]")
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.10")
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
            .getOrCreate()
        )

        # 2. Load Iris as a Spark DataFrame
        iris_csv_path = "../datasets/iris.csv"  # Adjust path to your actual CSV
        df_spark = cls.spark.read.csv(iris_csv_path, header=True, inferSchema=True)

        print("Spark DF columns:", df_spark.columns)

        # For demonstration, let's see if we have columns:
        #   "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"
        # We'll treat "Species" as the label that we index
        label_indexer = StringIndexer(inputCol="Species", outputCol="label").fit(df_spark)

        # 3. Vector Assemble for logistic regression
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

    def test_tabular_lime_iris(self):
        """
        Use DistributedTabularLIME on the Iris dataset
        and visualize the results for a small subset.
        """
        # 1. Create a small random background dataset
        background_data = broadcast(
            self.full_df.orderBy(rand()).limit(100).cache()
        )

        # 2. Build the DistributedTabularLIME
        numeric_features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        lime_explainer = DistributedTabularLIME(
            model=self.model,
            data=self.full_df,
            input_cols=numeric_features,
            target_col="probability",  # logistic regression outputs this vector
            target_classes=[1],        # let's explain class index=1
            num_samples=2000,
            background_data=background_data
        )

        # 3. Explain the subset (or entire dataset)
        lime_values = lime_explainer.explain(data=self.full_df)

        print("Global LIME values for numeric_features:", lime_values)

    def test_explain_row(self):
        """
        Test single-row LIME explanation using the DistributedTabularLIME approach.
        """
        # 1. Create a small background dataset
        background_data = broadcast(
            self.full_df.orderBy(rand()).limit(100).cache()
        )

        # 2. Initialize LIME explainer
        numeric_features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        lime_explainer = DistributedTabularLIME(
            model=self.model,
            data=self.full_df,  # Full dataset helps define the domain
            input_cols=numeric_features,
            target_col="probability",
            target_classes=[1],
            num_samples=2000,
            background_data=background_data
        )

        # 3. Explain a single row (or small subset)
        single_row_df = self.explain_instances.limit(1)  # or use self.full_df.limit(1)

        lime_values_single = lime_explainer.explain_row(data=single_row_df)
        print("LIME values for single row:")
        lime_values_single.show(truncate=False)


if __name__ == "__main__":
    unittest.main(argv=sys.argv, verbosity=2, exit=False)
