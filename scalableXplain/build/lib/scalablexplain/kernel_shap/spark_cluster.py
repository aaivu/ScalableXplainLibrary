from synapse.ml.explainers import TabularSHAP
from pyspark.sql.functions import rand, broadcast
import matplotlib.pyplot as plt
import numpy as np
import shap

class SparkKernelSHAPExplainer():
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

        self.explainer = TabularSHAP(
            inputCols=self.input_cols,
            outputCol="shapValues",
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
        Plot SHAP values for one or more instances.
        :param explained_df: Spark DataFrame returned by explain().
        :param original_df: Spark DataFrame of the original input used to explain.
        :param max_instances: Maximum instances to include in the plot.
        """
        if original_df is None:
            raise ValueError("original_df must be provided to retrieve feature values.")
        
        name = ""

        # 1) Collect SHAP values
        shap_pd = explained_df.limit(max_instances).select("shapValues").toPandas()
        if shap_pd.empty:
            raise ValueError("No SHAP values found in DataFrame.")

        shap_values = np.vstack(shap_pd["shapValues"])

        # 2) Drop the extra baseline column if present
        D = len(self.input_cols)
        if shap_values.shape[1] == D + 1:
            # baseline is first column
            shap_values = shap_values[:, 1:]

        # 3) Collect original features
        feature_pd = original_df.select(self.input_cols).limit(max_instances).toPandas()
        feature_values = feature_pd[self.input_cols].to_numpy()

        # 4) Final shape check
        if shap_values.shape != feature_values.shape:
            raise ValueError(
                f"Shape mismatch after dropping baseline: "
                f"SHAP shape={shap_values.shape}, Feature shape={feature_values.shape}"
            )

        # 5) Plot
        if shap_values.shape[0] == 1:
            name="single"
            shap.plots.bar(shap.Explanation(
                values=shap_values[0],
                data=feature_values[0],
                feature_names=self.input_cols
            ))
        else:
            name="multi"
            shap.plots.beeswarm(shap.Explanation(
                values=shap_values,
                data=feature_values,
                feature_names=self.input_cols
            ))
        # At the end of your plot method:
        plt.tight_layout()
        plt.savefig(f"shap_output_{name}.png")  # or .pdf/.svg
        print("[INFO] SHAP plot saved to shap_output.png")
        plt.show()
        plt.close()