import unittest
import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline

from scalableXplain.explainers.single_node.single_node_treeshap import TreeSHAPSingleNodeExplainer

class TestTreeSHAPSingleNodeExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder
            .appName("TestTreeSHAPSingleNodeExplainer")
            .master("local[*]")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_treeshap_explain_global(self):
        """
        Train a RandomForest and compute TreeSHAP global feature importance.
        """
        iris_csv_path = "../datasets/iris.csv"
        df_spark = self.spark.read.csv(iris_csv_path, header=True, inferSchema=True)

        label_indexer = StringIndexer(inputCol="Species", outputCol="indexedLabel")
        feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=3)
        pipeline = Pipeline(stages=[label_indexer, assembler, rf])
        model = pipeline.fit(df_spark)

        explainer = TreeSHAPSingleNodeExplainer(
            model=model.stages[-1],
            data=df_spark,
            feature_cols=feature_cols,
            label_col="Species",
            max_samples=100
        )

        explanation = explainer.explain()
        self.assertIsInstance(explanation, dict, "explain() should return a dictionary.")
        self.assertGreater(len(explanation), 0, "Explanation should contain feature importances.")

    # def test_treeshap_explain_row(self):
    #     """
    #     Compute SHAP values for a single row.
    #     """
    #     iris_csv_path = "../datasets/iris.csv"
    #     df_spark = self.spark.read.csv(iris_csv_path, header=True, inferSchema=True)
    #
    #     label_indexer = StringIndexer(inputCol="Species", outputCol="indexedLabel")
    #     feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    #     assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    #
    #     rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=3)
    #     pipeline = Pipeline(stages=[label_indexer, assembler, rf])
    #     model = pipeline.fit(df_spark)
    #
    #     explainer = TreeSHAPSingleNodeExplainer(
    #         model=model.stages[-1],
    #         data=df_spark,
    #         feature_cols=feature_cols,
    #         label_col="Species",
    #         max_samples=2
    #     )
    #
    #     single_row_df = df_spark.limit(150)
    #
    #     row_explanation = explainer.explain_row(single_row_df)
    #
    #     self.assertIsInstance(row_explanation, dict, "explain_row() should return a dictionary.")
    #     self.assertGreater(len(row_explanation), 0, "Explanation should contain SHAP values.")


if __name__ == "__main__":
    unittest.main(argv=sys.argv, verbosity=2, exit=False)
