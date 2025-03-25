import unittest
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from scalableXplain.explainers.single_node.feature_importance.single_node_kernelshap import SingleNodeKernelSHAP
from scalableXplain.explainers.single_node.feature_importance.single_node_treeshap import SingleNodeTreeSHAP
from scalableXplain.plots.bar_plot import BarPlot

class TestSingleNodeExplainers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the dataset from CSV
        iris_csv_path = "../datasets/iris.csv"
        data = pd.read_csv(iris_csv_path)

        # Define feature and target columns
        cls.feature_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        cls.X = data[cls.feature_columns]  # Select only the feature columns
        cls.y = data["Species"]  # Assuming "Species" is the target column

        # Split into train and test sets
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

        # Train a Decision Tree Classifier
        cls.model = DecisionTreeClassifier()
        cls.model.fit(cls.X_train, cls.y_train)

    def test_kernelshap_explain_global(self):
        """
        Compute KernelSHAP explanations for a dataset and plot feature importances.
        """
        explainer = SingleNodeKernelSHAP(model=self.model, data=self.X_train)
        explanation = explainer.explain(self.X_test[:5])

        # Check output type
        self.assertIsInstance(
            explanation,
            (list, np.ndarray),
            "explain() should return a list or NumPy array of SHAP values."
        )
        self.assertGreater(
            len(explanation),
            0,
            "Explanation should contain feature importances."
        )

        # Convert multi-class explanation to a single array if needed
        # For classification with multiple classes, shap_values is a list of arrays.
        if isinstance(explanation, list):
            # This merges all classes by averaging absolute SHAP values across classes
            explanation_array = np.mean(explanation, axis=0)  # shape: (n_samples, n_features)
        else:
            explanation_array = explanation  # shape: (n_samples, n_features)

        # Aggregate by mean absolute value across the samples
        mean_abs_shap = np.mean(np.abs(explanation_array), axis=0)
        # Create a dictionary of {feature_name: importance}
        feature_importances = dict(zip(self.feature_columns, mean_abs_shap))

        # Plot the SHAP values using BarPlot
        BarPlot(feature_importances).render("feature_importance_kernelshap.png")

    def test_kernelshap_explain_row(self):
        """
        Compute KernelSHAP explanation for a single row.
        """
        explainer = SingleNodeKernelSHAP(model=self.model, data=self.X_train)
        row_explanation = explainer.explain_row(self.X_test.iloc[0])

        # row_explanation can be an array or list-of-arrays (for multi-class)
        if isinstance(row_explanation, list):
            # Merge classes for test verification, e.g., by taking mean across classes
            row_explanation_array = np.mean(row_explanation, axis=0)
        else:
            row_explanation_array = row_explanation

        self.assertIsInstance(
            row_explanation_array,
            np.ndarray,
            "explain_row() should return a NumPy array (or a list of arrays for multi-class)."
        )
        self.assertGreater(
            len(row_explanation_array),
            0,
            "Row explanation should contain SHAP values."
        )

    def test_treeshap_explain_global(self):
        """
        Compute TreeSHAP explanations for a dataset.
        """
        explainer = SingleNodeTreeSHAP(self.model)
        explanation = explainer.explain(self.X_test[:5])

        # Check output type
        self.assertIsInstance(
            explanation,
            (list, np.ndarray),
            "explain() should return a list or NumPy array of SHAP values."
        )
        self.assertGreater(
            len(explanation),
            0,
            "Explanation should contain feature importances."
        )

        # Convert multi-class explanation to a single array if needed
        if isinstance(explanation, list):
            explanation_array = np.mean(explanation, axis=0)
        else:
            explanation_array = explanation

        # Aggregate by mean absolute value across the samples
        mean_abs_shap = np.mean(np.abs(explanation_array), axis=0)
        feature_importances = dict(zip(self.feature_columns, mean_abs_shap))

        # Optionally, plot TreeSHAP values as well
        # (We can reuse BarPlot for the aggregated shap values)
        BarPlot(feature_importances).render("feature_importance_treeshap.png")

    def test_treeshap_explain_row(self):
        """
        Compute TreeSHAP explanation for a single row.
        """
        explainer = SingleNodeTreeSHAP(self.model)
        row_explanation = explainer.explain_row(self.X_test.iloc[0])

        if isinstance(row_explanation, list):
            row_explanation_array = np.mean(row_explanation, axis=0)
        else:
            row_explanation_array = row_explanation

        self.assertIsInstance(
            row_explanation_array,
            np.ndarray,
            "explain_row() should return a NumPy array (or a list of arrays for multi-class)."
        )
        self.assertGreater(
            len(row_explanation_array),
            0,
            "Row explanation should contain SHAP values."
        )

if __name__ == "__main__":
    unittest.main(argv=sys.argv, verbosity=2, exit=False)
