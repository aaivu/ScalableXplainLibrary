# import unittest
# import sys
# from pyspark.sql import SparkSession
# from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml.feature import VectorAssembler, StringIndexer
# from pyspark.ml import Pipeline
#
# from scalableXplain.explainers.single_node.feature_importance.single_node_treeshap import TreeSHAPSingleNodeExplainer
# from scalableXplain.plots.bar_plot import BarPlot
#
#
# class TestTreeSHAPSingleNodeExplainer(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.spark = (
#             SparkSession.builder
#             .appName("TestTreeSHAPSingleNodeExplainer")
#             .master("local[*]")
#             .getOrCreate()
#         )
#
#     @classmethod
#     def tearDownClass(cls):
#         cls.spark.stop()
#
#     def test_treeshap_explain_global(self):
#         """
#         Train a RandomForest and compute TreeSHAP global feature importance.
#         """
#         iris_csv_path = "../datasets/iris.csv"
#         df_spark = self.spark.read.csv(iris_csv_path, header=True, inferSchema=True)
#
#         label_indexer = StringIndexer(inputCol="Species", outputCol="indexedLabel")
#         feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
#         assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
#
#         rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=3)
#         pipeline = Pipeline(stages=[label_indexer, assembler, rf])
#         model = pipeline.fit(df_spark)
#
#         explainer = TreeSHAPSingleNodeExplainer(
#             model=model.stages[-1],
#             data=df_spark,
#             feature_cols=feature_cols,
#             label_col="Species",
#             max_samples=100
#         )
#
#         explanation = explainer.explain()
#         self.assertIsInstance(explanation, dict, "explain() should return a dictionary.")
#         self.assertGreater(len(explanation), 0, "Explanation should contain feature importances.")
#         BarPlot(explanation).render("global_feature_importance.png")
#
#     def test_treeshap_explain_row(self):
#         """
#         Compute SHAP values for a single row.
#         """
#         iris_csv_path = "../datasets/iris.csv"
#         df_spark = self.spark.read.csv(iris_csv_path, header=True, inferSchema=True)
#
#         label_indexer = StringIndexer(inputCol="Species", outputCol="indexedLabel")
#         feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
#         assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
#
#         rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=3)
#         pipeline = Pipeline(stages=[label_indexer, assembler, rf])
#         model = pipeline.fit(df_spark)
#
#         explainer = TreeSHAPSingleNodeExplainer(
#             model=model.stages[-1],
#             data=df_spark,
#             feature_cols=feature_cols,
#             label_col="Species",
#             max_samples=2
#         )
#
#         single_row_df = df_spark.limit(3)
#
#         row_explanation = explainer.explain_row(single_row_df)
#
#         self.assertIsInstance(row_explanation, dict, "explain_row() should return a dictionary.")
#         self.assertGreater(len(row_explanation), 0, "Explanation should contain SHAP values.")
#
# if __name__ == "__main__":
#     unittest.main(argv=sys.argv, verbosity=2, exit=False)

import unittest
import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline

from scalableXplain.explainers.single_node.feature_importance.single_node_treeshap import TreeSHAPSingleNodeExplainer
from scalableXplain.plots.bar_plot import BarPlot


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
        iris_csv_path = "../datasets/iris.csv"  # Update if needed.  Make sure this path is correct.
        df_spark = self.spark.read.csv(iris_csv_path, header=True, inferSchema=True)

        # Index the label
        label_indexer = StringIndexer(inputCol="Species", outputCol="indexedLabel")
        feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        rf = RandomForestClassifier(
            labelCol="indexedLabel",
            featuresCol="features",
            numTrees=50
        )
        pipeline = Pipeline(stages=[label_indexer, assembler, rf])
        model = pipeline.fit(df_spark)

        # Create explainer, pulling the final RF model from pipeline
        explainer = TreeSHAPSingleNodeExplainer(
            model=model.stages[-1],
            data=df_spark,
            feature_cols=feature_cols,
            label_col="Species",
            max_samples=100  # sample for global explanation
        )

        # Run global explanation
        explanation = explainer.explain()

        self.assertIsInstance(explanation, dict, "explain() should return a dictionary.")
        self.assertGreater(len(explanation), 0, "Explanation should contain feature importances.")
        # Check that the explanation dict covers our feature columns
        self.assertSetEqual(set(explanation.keys()), set(feature_cols),
                            "Explanation should return SHAP values for each feature.")

        # Optionally plot the results
        # BarPlot(explanation).render("global_feature_importanceee.png") # Removed the render.

    def test_treeshap_explain_row(self):
        """
        Compute SHAP values for a single row.
        """
        iris_csv_path = "../datasets/iris.csv"  # Update the path if needed.
        df_spark = self.spark.read.csv(iris_csv_path, header=True, inferSchema=True)

        # Index the label
        label_indexer = StringIndexer(inputCol="Species", outputCol="indexedLabel")
        feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        rf = RandomForestClassifier(
            labelCol="indexedLabel",
            featuresCol="features",
            numTrees=3
        )
        pipeline = Pipeline(stages=[label_indexer, assembler, rf])
        model = pipeline.fit(df_spark)

        explainer = TreeSHAPSingleNodeExplainer(
            model=model.stages[-1],
            data=df_spark,
            feature_cols=feature_cols,
            label_col="Species",
            max_samples=2  # used only for global explanation sampling
        )

        # Select a single row
        single_row_df = df_spark.limit(1)

        # Run row-level explanation
        row_explanation = explainer.explain_row(single_row_df)

        self.assertIsInstance(row_explanation, dict, "explain_row() should return a dictionary.")
        self.assertGreater(len(row_explanation), 0, "Explanation should contain SHAP values.")

        # Verify each feature is present; SHAP value may be float or list (multi-class)
        self.assertSetEqual(set(row_explanation.keys()), set(feature_cols),
                            "Row-level explanation should contain SHAP values for each feature.")

        for feature, shap_value in row_explanation.items():
            self.assertIsInstance(
                shap_value,
                (float, list),
                f"SHAP value for {feature} should be float (single-class) or list (multi-class)."
            )



if __name__ == "__main__":
    unittest.main(argv=sys.argv, verbosity=2, exit=False)