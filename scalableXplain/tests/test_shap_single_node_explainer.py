import unittest
import sys

import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession

# Import from your package:
from scalableXplain.explainers.single_node.single_node_shap import SHAPSingleNodeExplainer
from scalableXplain.explanations.feature_importance_explanation import FeatureImportanceExplanation


class TestIrisShapWithScalableXplain(unittest.TestCase):
    """
    Demonstrates using the scalableXplain package to explain a scikit-learn
    logistic regression model trained on the Iris dataset, loaded via PySpark.
    """

    @classmethod
    def setUpClass(cls):
        # Create a Spark session once for all tests in this class
        cls.spark = (SparkSession.builder
                     .appName("TestIrisShapWithScalableXplain")
                     .master("local[*]")
                     .getOrCreate())

    @classmethod
    def tearDownClass(cls):
        # Stop the Spark session after all tests
        cls.spark.stop()

    def test_iris_shap_explainer(self):
        # 1. Path to your iris.csv (adjust as needed)
        iris_csv_path = "../datasets/iris.csv"

        # 2. Load iris.csv into a Spark DataFrame
        iris_spark_df = self.spark.read.csv(iris_csv_path, header=True, inferSchema=True)
        # self.assertGreater(iris_spark_df.count(), 0, "Expected non-empty Spark DataFrame for iris data.")

        # 3. Convert Spark DataFrame to pandas for inspection
        iris_df = iris_spark_df.toPandas()
        print("Columns in iris_df:", iris_df.columns.tolist())

        # 4. Separate feature columns and label column
        feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        label_col = "Species"

        # Validate that these columns exist in the DataFrame
        for col in feature_cols + [label_col]:
            self.assertIn(col, iris_df.columns, f"Column '{col}' not found in Iris CSV")

        X = iris_df[feature_cols]
        y = iris_df[label_col]

        # 5. Encode the string label (species) into numeric
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # 6. Train logistic regression in single-node scikit-learn
        #    (multinomial for the 3-class Iris problem)
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
        model.fit(X, y_encoded)

        # 7. Instantiate the SHAPSingleNodeExplainer from your package
        explainer = SHAPSingleNodeExplainer(
            model=model,
            data=iris_spark_df,  # pass Spark DataFrame
            feature_cols=feature_cols,
            label_col=label_col,
            max_samples=150  # iris dataset has 150 rows; no real sampling needed
        )

        # 8. Generate a global explanation
        explanation = explainer.explain()
        self.assertIsInstance(
            explanation,
            FeatureImportanceExplanation,
            "explain() should return a FeatureImportanceExplanation instance"
        )
        print(f"Global SHAP-based feature importance:\n{explanation}")

        # 9. (Optional) Explain a single row
        single_row_df = iris_spark_df.limit(1)  # Spark DF with just one row
        row_explanation = explainer.explain_row(single_row_df)
        self.assertIsInstance(
            row_explanation,
            FeatureImportanceExplanation,
            "explain_row() should return a FeatureImportanceExplanation instance"
        )
        print(f"Row-level SHAP-based explanation:\n{row_explanation}")

        # 10. (Optional) Create and save a force plot for the single row:
        # Convert single_row_df to pandas for the actual shap force plot
        single_row_pd = single_row_df.toPandas()[feature_cols]
        shap_values = shap.Explainer(model, X, feature_names=feature_cols)(single_row_pd)
        # For multi-class, shap_values.values shape: [n_rows, n_features, n_classes]
        sample_idx = 0
        class_idx = 0  # e.g., SHAP for the first class
        force_plot_html = shap.force_plot(
            base_value=shap_values.base_values[sample_idx, class_idx],
            shap_values=shap_values.values[sample_idx, :, class_idx],
            features=single_row_pd.iloc[sample_idx],
            feature_names=feature_cols,
            matplotlib=False
        )
        shap.save_html("test_iris_shap_force_plot.html", force_plot_html)
        print("Saved SHAP force plot to test_iris_shap_force_plot.html")


if __name__ == "__main__":
    unittest.main(argv=sys.argv, verbosity=2, exit=False)
