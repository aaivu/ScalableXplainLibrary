import unittest
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Replace with your actual import path for SingleNodeLIME
from scalableXplain.explainers.single_node.feature_importance.single_node_lime import SingleNodeLIME
from scalableXplain.plots.bar_plot import BarPlot

class TestSingleNodeLIME(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the dataset from CSV or use a built-in dataset
        # For consistency, let's assume you have iris.csv with columns:
        # ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
        iris_csv_path = "../datasets/iris.csv"
        data = pd.read_csv(iris_csv_path)

        # Define feature and target columns
        cls.feature_columns = [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
        ]
        cls.X = data[cls.feature_columns]
        cls.y = data["Species"]

        # Train/test split
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

        # Train a simple Decision Tree Classifier
        cls.model = DecisionTreeClassifier()
        cls.model.fit(cls.X_train, cls.y_train)

    def test_lime_explain_global(self):
        """
        Compute LIME explanations for *all* test samples and aggregate
        their local coefficients to get a pseudo-'global' view of feature importance.
        """
        # Initialize SingleNodeLIME
        explainer = SingleNodeLIME(
            model=self.model,
            data=self.X_train,
            feature_names=self.feature_columns
        )

        # Explain each row in the test set
        all_explanations = []
        for i in range(len(self.X_test)):
            row = self.X_test.iloc[i]
            # explain_row() is a convenience method in your SingleNodeLIME for a single row
            exp = explainer.explain_row(row, num_features=4, labels=(0,))
            all_explanations.append(exp)

        self.assertEqual(
            len(all_explanations),
            len(self.X_test),
            "Should return one explanation object per test sample."
        )

        # We'll collect the *absolute* local coefficients for each feature
        # to approximate how often & how strongly each feature influences predictions.
        aggregated_importances = {f: 0.0 for f in self.feature_columns}

        for exp in all_explanations:
            local_importances = exp.as_list(label=0)
            # local_importances = [(feature_desc, weight), ...]
            for feat_desc, weight in local_importances:
                # Check which feature name is present in the descriptor
                for feat_name in self.feature_columns:
                    if feat_name in feat_desc:
                        aggregated_importances[feat_name] += abs(weight)
                        break

        # Ensure at least one feature has a non-zero aggregated weight
        max_imp = max(aggregated_importances.values())
        self.assertGreater(
            max_imp,
            0.0,
            "At least one feature should have a non-zero importance across the test set."
        )

        # Plot the aggregated importances
        BarPlot(aggregated_importances).render("feature_importance_lime_global.png")

    def test_lime_explain_single_row_and_plot(self):
        """
        Demonstrates computing and plotting the explanation of a single row.
        """
        explainer = SingleNodeLIME(
            model=self.model,
            data=self.X_train,
            feature_names=self.feature_columns
        )

        # Let’s pick a single test row (e.g., index=0)
        single_row = self.X_test.iloc[0]
        exp = explainer.explain_row(single_row, num_features=4, labels=(0,))

        # Basic check that it returns a LIME explanation object
        explanation_list = exp.as_list(label=0)
        self.assertTrue(len(explanation_list) > 0, "Explanation should have at least one feature.")

        # Turn the explanation into a feature->weight dictionary
        # (Here we're just capturing the raw weight, but you could use abs(weight).)
        row_importances = {}
        for feat_desc, weight in explanation_list:
            for feat_name in self.feature_columns:
                if feat_name in feat_desc:
                    row_importances[feat_name] = weight
                    break  # found matching feature

        # Plot the single-row explanation
        BarPlot(row_importances).render("lime_single_row_explanation.png")

        # Optionally test that at least one feature has a non-zero weight
        non_zero_features = [f for f, w in row_importances.items() if abs(w) > 0]
        self.assertTrue(
            len(non_zero_features) > 0,
            "Expected at least one non-zero feature weight for the single-row explanation."
        )


if __name__ == "__main__":
    unittest.main(argv=sys.argv, verbosity=2, exit=False)
