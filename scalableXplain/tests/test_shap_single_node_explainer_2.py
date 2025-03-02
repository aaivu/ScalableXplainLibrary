import unittest
import sys
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
import pandas as pd

# scikit-learn for local training
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Import classes from your package
from scalableXplain.explainers.single_node.single_node_shap import SHAPSingleNodeExplainer
from scalableXplain.explanations.feature_importance_explanation import FeatureImportanceExplanation


class TestSHAPSingleNodeExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create Spark Session once for this test class
        cls.spark = (SparkSession.builder
                     .appName("TestSHAPSingleNodeExplainer_Iris")
                     .master("local[*]")
                     .getOrCreate())

    @classmethod
    def tearDownClass(cls):
        # Stop the Spark Session
        cls.spark.stop()

    def test_iris_explain_global(self):
        """
        Test Case: Use the Iris dataset to verify that SHAPSingleNodeExplainer's
        `explain()` method returns a valid global FeatureImportanceExplanation.
        """
        # 1. Adjust path to your local iris CSV file
        iris_csv_path = "../datasets/iris.csv"

        # 2. Read iris.csv into a Spark DataFrame
        iris_spark_df = self.spark.read.csv(
            iris_csv_path, header=True, inferSchema=True
        )
        row_count = iris_spark_df.count()
        self.assertGreater(row_count, 0, "Iris Spark DataFrame should not be empty.")

        # 3. Convert Spark DataFrame to pandas for feature/label extraction
        iris_df = iris_spark_df.toPandas()
        print("Iris CSV columns detected:", iris_df.columns.tolist())
        # Example: ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

        # 4. Define your feature and label columns (adjust to match your CSV exactly!)
        feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        label_col = "Species"

        # 5. Quick check that these columns exist in the DataFrame
        for col in feature_cols + [label_col]:
            self.assertIn(col, iris_df.columns, f"Column '{col}' not found in Iris DataFrame")

        # 6. Separate features (X) and label (y)
        X = iris_df[feature_cols]
        y = iris_df[label_col]

        # 7. Encode the Species label into numeric for logistic regression
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # 8. Train a local scikit-learn Logistic Regression model
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)
        model.fit(X, y_encoded)

        # 9. Instantiate the SHAPSingleNodeExplainer with the Spark DF
        explainer = SHAPSingleNodeExplainer(
            model=model,
            data=iris_spark_df,  # pass the Spark DF, single-node SHAP will convert internally
            feature_cols=feature_cols,
            label_col=label_col,
            max_samples=row_count  # avoid sampling for the full dataset
        )

        # 10. Call explain() to get a global FeatureImportanceExplanation
        explanation = explainer.explain()

        # 11. Verify the output
        self.assertIsInstance(
            explanation, FeatureImportanceExplanation,
            "explain() should return a FeatureImportanceExplanation instance"
        )
        self.assertEqual(
            explanation.method, "SHAP",
            "Expected method='SHAP' in the explanation"
        )
        # Check that each feature has an importance value
        for col in feature_cols:
            self.assertIn(col, explanation.feature_importances,
                          f"Missing SHAP importance for feature '{col}'")

        # 12. Print or log the explanation
        print("\nGlobal SHAP explanation for Iris dataset:")
        print(explanation)



if __name__ == "__main__":
    unittest.main(argv=sys.argv, verbosity=2, exit=False)
