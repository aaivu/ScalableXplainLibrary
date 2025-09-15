"""
SparkLIMEExplainer provides a wrapper for SynapseML's TabularLIME to interpret predictions of Spark ML models using LIME values.
It supports explanation of both single and multiple instances, visualization of feature importances, and integration with PySpark DataFrames.
Attributes:
    model: Trained Spark ML model (compatible with SynapseML TabularLIME).
    input_cols: List of input feature column names.
    target_col: Name of the output column to explain (default: "probability").
    target_classes: List of target class indices for explanation (default: [1]).
    background_data: Spark DataFrame used as the background distribution for LIME.
    num_samples: Number of samples used for LIME approximation.
    explainer: TabularLIME instance (initialized after build_explainer is called).
Methods:
    __init__(model, input_cols, target_col="probability", target_classes=[1], background_data=None, num_samples=5000):
        Initializes the SparkLIMEExplainer with the given model, input columns, target column, target classes,
        optional background data, and number of samples for LIME approximation.
    build_explainer(training_data):
        Initializes the TabularLIME explainer using the provided training data. If background_data is not set,
        samples 100 rows from training_data as the background.
    explain(instances_df):
        Computes LIME values for the provided instances. Returns a Spark DataFrame with LIME values for each instance.
    plot(explained_df, original_df, max_instances=100):
        Visualizes LIME values using bar or beeswarm plots. Saves the plot as a PNG file and displays it.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from synapse.ml.explainers import TabularLIME
from pyspark.sql.functions import rand, broadcast
import numpy as np
import pandas as pd

class SparkLIMEExplainer():
    def __init__(self, model, input_cols, target_col="probability", target_classes=[1], background_data=None, num_samples=5000):
        self.model = model
        self.input_cols = input_cols
        self.target_col = target_col
        self.target_classes = target_classes
        self.num_samples = num_samples
        self.background_data = background_data
        self.explainer = None

    def build_explainer(self, training_data):
        if self.background_data is None:
            self.background_data = training_data.orderBy(rand()).limit(100).cache()

        self.explainer = TabularLIME(
            inputCols=self.input_cols,
            outputCol="limeValues",
            numSamples=self.num_samples,
            model=self.model,
            targetCol=self.target_col,
            targetClasses=self.target_classes,
            backgroundData=broadcast(self.background_data),
        )

    def explain(self, instances_df):
        if self.explainer is None:
            raise ValueError("Explainer has not been built. Call build_explainer() first.")
        return self.explainer.transform(instances_df)

    def plot(self, explained_df, original_df, max_instances=100):
        """
        Plot LIME values for one or more instances using bar plot (single) or Seaborn beeswarm (multi).
        """
        if original_df is None:
            raise ValueError("original_df must be provided.")

        name = ""

        lime_pd = explained_df.limit(max_instances).select("limeValues").toPandas()
        if lime_pd.empty:
            raise ValueError("No limeValues found in DataFrame.")

        lime_values = np.vstack(lime_pd["limeValues"])

        feature_pd = original_df.select(self.input_cols).limit(max_instances).toPandas()
        feature_values = feature_pd[self.input_cols].to_numpy()

        if lime_values.shape != feature_values.shape:
            raise ValueError(
                f"Shape mismatch: LIME shape={lime_values.shape}, Feature shape={feature_values.shape}"
            )

        # --------- Plot logic ----------
        if lime_values.shape[0] == 1:
            name = "single"
            values = lime_values[0]
            data = feature_values[0]

            sorted_idx = np.argsort(np.abs(values))[::-1]
            topk = sorted_idx[:15]  # top 15 features

            plt.figure(figsize=(8, 6))
            plt.barh(np.array(self.input_cols)[topk], values[topk], color="skyblue")
            plt.gca().invert_yaxis()
            plt.xlabel("LIME Value")
            plt.title("LIME Explanation (Single Instance)")

        else:
            name = "multi"
            avg_vals = np.mean(np.abs(lime_values), axis=0)
            sorted_idx = np.argsort(avg_vals)[::-1]
            topk = sorted_idx[:15]

            lime_topk = lime_values[:, topk]
            feature_names_topk = [self.input_cols[i] for i in topk]

            melted = pd.DataFrame(lime_topk, columns=feature_names_topk)
            melted = melted.melt(var_name="Feature", value_name="LIME Value")

            plt.figure(figsize=(10, 6))
            sns.swarmplot(data=melted, x="LIME Value", y="Feature", orient="h", size=4)
            plt.title("LIME Beeswarm Plot (Top Features)")
            plt.tight_layout()

        filename = f"lime_output_{name}.png"
        plt.savefig(filename)
        print(f"[INFO] LIME plot saved to {filename}")
        plt.close()