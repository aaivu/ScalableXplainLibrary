# kernel_shap/single_node.py

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    def plot(self, max_instances=100):
        """
        Plots SHAP values using bar or beeswarm plots depending on the number of instances.
        """
        if not hasattr(self, 'last_shap_values'):
            raise ValueError("No SHAP values found. Run explain() first.")
        
        if len(self.last_instances) == 1:
            shap.plots.bar(shap.Explanation(values=self.last_shap_values[0], 
                                             data=self.last_instances.iloc[0].values, 
                                             feature_names=self.input_cols))
        else:
            sample_idx = self.last_instances.index[:max_instances]
            shap.plots.beeswarm(
                shap.Explanation(
                    values=self.last_shap_values[:max_instances],
                    data=self.last_instances.loc[sample_idx].values,
                    feature_names=self.input_cols
                )
            )
        plt.show()