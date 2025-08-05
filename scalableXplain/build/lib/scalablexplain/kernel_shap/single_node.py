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
        return pd.DataFrame(shap_values, columns=self.input_cols, index=instances_df.index)

    def plot(self, shap_values_df: pd.DataFrame, instances_df: pd.DataFrame, max_instances: int = 100):
        """
        Plot SHAP values using bar or beeswarm plot.
        
        :param shap_values_df: DataFrame of SHAP values (output of explain()).
        :param instances_df: DataFrame of original input data.
        :param max_instances: Max number of instances to plot.
        """
        if shap_values_df is None or shap_values_df.empty:
            raise ValueError("shap_values_df is empty or None.")
        if instances_df is None or instances_df.empty:
            raise ValueError("instances_df is empty or None.")

        name = "single" if shap_values_df.shape[0] == 1 else "multi"
        values = shap_values_df.values
        data = instances_df[self.input_cols].values

        if name == "single":
            shap.plots.bar(shap.Explanation(
                values=values[0],
                data=data[0],
                feature_names=self.input_cols
            ))
        else:
            shap.plots.beeswarm(shap.Explanation(
                values=values[:max_instances],
                data=data[:max_instances],
                feature_names=self.input_cols
            ))

        plt.tight_layout()
        plt.savefig(f"shap_output_{name}.png")
        print(f"[INFO] SHAP plot saved to shap_output_{name}.png")
        plt.show()
        plt.close()
