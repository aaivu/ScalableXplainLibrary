import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer


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

        for i in range(len(instance_np)):
            exp = self.explainer.explain_instance(
                instance_np[i],
                self.model.predict_proba if self.mode == "classification" else self.model.predict,
                num_features=len(self.input_cols),
                num_samples=self.num_samples
            )
            label = self.target_classes[0] if self.mode == "classification" else None
            all_feature_names.update(dict(exp.as_list(label=label)).keys())

        all_feature_names = sorted(all_feature_names)

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

    def plot(self, lime_df, original_df, max_instances=100):
        """
        Plot LIME values using either bar (single instance) or seaborn beeswarm (multiple).

        Parameters:
        - lime_df: pandas DataFrame with LIME values (output from explain)
        - original_df: pandas DataFrame of original features corresponding to lime_df
        - max_instances: Maximum number of instances to visualize
        """
        if lime_df is None or lime_df.empty:
            raise ValueError("LIME DataFrame is empty.")

        lime_df = lime_df.head(max_instances)
        original_df = original_df[self.input_cols].head(max_instances)

        name = ""

        if lime_df.shape[0] == 1:
            name = "single"
            row = lime_df.iloc[0]
            sorted_idx = row.abs().sort_values(ascending=False).index[:15]
            values = row[sorted_idx]

            plt.figure(figsize=(8, 6))
            plt.barh(sorted_idx, values, color="skyblue")
            plt.gca().invert_yaxis()
            plt.xlabel("LIME Value")
            plt.title("LIME Explanation (Single Instance)")

        else:
            name = "multi"
            top_features = (
                lime_df.abs().mean(axis=0)
                .sort_values(ascending=False)
                .head(15)
                .index.tolist()
            )
            melted = lime_df[top_features].copy()
            melted["Instance"] = range(len(melted))
            melted = melted.melt(id_vars="Instance", var_name="Feature", value_name="LIME Value")

            plt.figure(figsize=(10, 6))
            sns.swarmplot(data=melted, x="LIME Value", y="Feature", orient="h", size=4)
            plt.title("LIME Beeswarm Plot (Top Features)")
            plt.tight_layout()

        filename = f"lime_output_{name}.png"
        plt.savefig(filename)
        print(f"[INFO] LIME plot saved to {filename}")
        plt.close()
