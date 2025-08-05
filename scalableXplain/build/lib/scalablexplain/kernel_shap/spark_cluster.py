# kernel_shap/single_node.py

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KernelSHAPExplainer:
    def __init__(self, model, input_cols, background_data=None, num_samples=5000):
        self.model = model
        self.input_cols = input_cols
        self.num_samples = num_samples
        self.background_data = background_data
        self.explainer = None
        self.last_shap_values = None
        self.last_instances = None

    def build_explainer(self, training_data):
        if self.background_data is None:
            self.background_data = training_data[self.input_cols].sample(n=100, random_state=42)

        def predict_fn(X):
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)[:, 1]
            return self.model.predict(X)

        self.explainer = shap.KernelExplainer(predict_fn, self.background_data)

    def explain(self, instances_df):
        if self.explainer is None:
            raise ValueError("Explainer has not been built. Call build_explainer() first.")

        X = instances_df[self.input_cols].values
        shap_values = self.explainer.shap_values(X, nsamples=self.num_samples)

        self.last_shap_values = np.array(shap_values)
        self.last_instances = instances_df[self.input_cols]
        return pd.DataFrame(shap_values, columns=self.input_cols, index=instances_df.index)

    def plot(self, max_instances=100):
        """
        Plots SHAP values using bar (if one instance) or beeswarm (if multiple), matching SparkKernelSHAPExplainer.
        """
        if self.last_shap_values is None or self.last_instances is None:
            raise ValueError("Run explain() first before plotting.")

        shap_values = self.last_shap_values
        feature_values = self.last_instances.values
        name = ""

        if shap_values.shape[0] == 1:
            name = "single"
            shap.plots.bar(shap.Explanation(
                values=shap_values[0],
                data=feature_values[0],
                feature_names=self.input_cols
            ))
        else:
            name = "multi"
            max_instances = min(max_instances, shap_values.shape[0])
            shap.plots.beeswarm(shap.Explanation(
                values=shap_values[:max_instances],
                data=feature_values[:max_instances],
                feature_names=self.input_cols
            ))

        plt.tight_layout()
        plt.savefig(f"shap_output_{name}.png")
        print(f"[INFO] SHAP plot saved to shap_output_{name}.png")
        plt.show()
        plt.close()
