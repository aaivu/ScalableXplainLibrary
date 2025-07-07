# lime/single_node.py

from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd

class LIMEExplainer:
    def __init__(self, model, input_cols, class_names=None, mode="classification",
                 target_classes=[1], background_data=None, num_samples=5000):
        """
        model: Trained scikit-learn model (should implement predict_proba or predict)
        input_cols: List of column names in the input data
        class_names: List of class labels (for classification)
        mode: "classification" or "regression"
        target_classes: List of class indices to explain (only for classification)
        background_data: Optional pandas DataFrame for training the explainer
        num_samples: Number of samples LIME uses to generate explanations
        """
        self.model = model
        self.input_cols = input_cols
        self.class_names = class_names
        self.mode = mode
        self.target_classes = target_classes
        self.num_samples = num_samples
        self.background_data = background_data
        self.explainer = None

    def build_explainer(self, training_data):
        """
        Prepares the LimeTabularExplainer using background data or training data.
        """
        if self.background_data is None:
            self.background_data = training_data.sample(n=100, random_state=42)

        background_np = self.background_data[self.input_cols].to_numpy()

        self.explainer = LimeTabularExplainer(
            training_data=background_np,
            feature_names=self.input_cols,
            class_names=self.class_names,
            mode=self.mode,
            discretize_continuous=False,
            sample_around_instance=True
        )

    def explain(self, instances_df):
        """
        Returns a DataFrame of shape (n_instances, n_features_selected).
        Each row contains the LIME explanation (weights) for that instance.
        Columns are the LIME feature conditions (e.g., 'x1 <= 2.25').
        Missing features (not selected for a sample) are filled with 0.0.
        """
        if self.explainer is None:
            raise ValueError("Explainer has not been built. Call build_explainer() first.")

        instance_np = instances_df[self.input_cols].to_numpy()
        explanations = []

        all_feature_names = set()

        # First pass: collect all possible feature names across all explanations
        for i in range(len(instance_np)):
            exp = self.explainer.explain_instance(
                instance_np[i],
                self.model.predict_proba if self.mode == "classification" else self.model.predict,
                num_features=len(self.input_cols),
                num_samples=self.num_samples
            )
            label = self.target_classes[0] if self.mode == "classification" else None
            all_feature_names.update(dict(exp.as_list(label=label)).keys())

        all_feature_names = sorted(all_feature_names)  # Optional: consistent column order

        # Second pass: build one-hot weight dictionary per instance
        for i in range(len(instance_np)):
            exp = self.explainer.explain_instance(
                instance_np[i],
                self.model.predict_proba if self.mode == "classification" else self.model.predict,
                num_features=len(self.input_cols),
                num_samples=self.num_samples
            )
            label = self.target_classes[0] if self.mode == "classification" else None
            explanation_dict = dict(exp.as_list(label=label))

            row = {feature: explanation_dict.get(feature, 0.0) for feature in all_feature_names}
            explanations.append(row)

        return pd.DataFrame(explanations)
