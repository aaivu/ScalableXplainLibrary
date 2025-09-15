'''
KernelSHAPExplainer provides a wrapper for SHAP's KernelExplainer to interpret predictions of 
sklearn-compatible models using SHAP values. It supports both regression and classification models, 
automatically selecting the appropriate prediction function. The class allows users to build an explainer 
from training data, generate SHAP values for new instances, and visualize feature attributions.
Attributes:
    model: Trained sklearn-compatible model (should implement predict_proba or predict).
    input_cols: List of input feature column names.
    background_data: DataFrame or numpy array to use as the background distribution for SHAP.
    num_samples: Number of samples used for KernelSHAP approximation.
    explainer: SHAP KernelExplainer instance (initialized after build_explainer is called).
Methods:
    __init__(model, input_cols, background_data=None, num_samples=5000):
        Initializes the KernelSHAPExplainer with the given model, input columns, optional background data, 
        and number of samples for SHAP approximation.
    build_explainer(training_data):
        Initializes the SHAP KernelExplainer using the provided training data. If background_data is not set, 
        samples 100 rows from training_data as the background. Selects the appropriate prediction function 
        based on model type.
    explain(instances_df):
        Computes SHAP values for the provided instances. Returns a DataFrame of SHAP values with feature 
        columns and instance indices.
    plot(shap_values_df, instances_df, max_instances=100):
        Visualizes SHAP values using bar or beeswarm plots. Saves the plot as a PNG file and displays it.
'''

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
        Initializes the SHAP KernelExplainer for the model using the provided training data.

        Args:
            training_data (pd.DataFrame): The dataset used to sample background data for the explainer. 
                Should contain columns specified in self.input_cols.

        Functionality:
            - Samples 100 rows from the training data to use as background data if not already set.
            - Defines a prediction function compatible with SHAP, using either `predict_proba` or `predict` depending on regression or classification use case.
            - Instantiates a SHAP KernelExplainer with the prediction function and background data.

        """
        I
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
