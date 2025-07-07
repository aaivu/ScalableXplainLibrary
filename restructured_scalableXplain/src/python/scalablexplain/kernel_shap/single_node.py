# kernel_shap/single_node.py

import shap
import numpy as np
import pandas as pd

class KernelSHAPExplainer:
    def __init__(self, model, input_cols, background_data=None, num_samples=5000):
        """
        model: Trained sklearn-compatible model (should implement predict_proba or predict)
        input_cols: List of input feature column names
        background_data: DataFrame or numpy array to use as the background distribution
        num_samples: Number of samples used for KernelSHAP approximation
        """
        self.model = model
        self.input_cols = input_cols
        self.num_samples = num_samples
        self.background_data = background_data
        self.explainer = None

    def build_explainer(self, training_data):
        """
        Prepares the KernelExplainer using a subset of the training data.
        """
        if self.background_data is None:
            # Default to using 100 background samples
            self.background_data = training_data[self.input_cols].sample(n=100, random_state=42)

        def predict_fn(X):
            # Use predict_proba if available; fallback to predict
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)[:, 1]
            return self.model.predict(X)

        self.explainer = shap.KernelExplainer(predict_fn, self.background_data)

    def explain(self, instances_df):
        """
        Returns a matrix of SHAP values for the given instances.
        """
        if self.explainer is None:
            raise ValueError("Explainer has not been built. Call build_explainer() first.")

        X = instances_df[self.input_cols].values
        shap_values = self.explainer.shap_values(X, nsamples=self.num_samples)
        print(type(shap_values))
        return pd.DataFrame(shap_values, columns=self.input_cols, index=instances_df.index)
