"""
SHAP-based Single-Node Explainer for PySpark, with a bar plot of global importances.

This version extends the original code by producing a Matplotlib bar plot
of mean absolute SHAP values in the `explain()` method.
"""

from typing import Optional, Any
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame as SparkDataFrame

from scalableXplain.explainers.single_node.single_node_explainer import SingleNodeExplainer
from scalableXplain.explanations.feature_importance_explanation import FeatureImportanceExplanation


class SHAPSingleNodeExplainer(SingleNodeExplainer):
    """
    A single-node explainer that uses SHAP to compute feature importances.

    This class is responsible for:
      - Converting PySpark data into a local (pandas) format.
      - Running SHAP explanations on the local machine.
      - Returning a FeatureImportanceExplanation object (or any specialized
        SHAP-based explanation structure).
      - Plotting a bar plot of the global (mean absolute) SHAP values.
    """

    def __init__(self,
                 model: Any,
                 data: SparkDataFrame,
                 feature_cols: Optional[list[str]] = None,
                 label_col: Optional[str] = None,
                 max_samples: int = 1000):
        """
        Parameters
        ----------
        model : Any
            The trained (single-node friendly) model to explain.
            Should have a predict() or predict_proba() method.
        data : SparkDataFrame
            The dataset in Spark format. Will be converted to pandas for SHAP.
        feature_cols : list of str, optional
            Names of the feature columns to be used by SHAP. If None, all columns
            except label_col are used.
        label_col : str, optional
            Name of the label column. If provided, it will be excluded from the
            feature columns in the explanation.
        max_samples : int, optional
            Maximum number of rows to sample from the Spark DataFrame to avoid
            excessive memory usage in single-node mode.
        """
        super().__init__(model=model, data=data)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.max_samples = max_samples

    def explain(self) -> FeatureImportanceExplanation:
        """
        Generates a SHAP-based feature-importance explanation for the entire dataset
        and produces a bar plot (saved as "shap_bar_plot.png") showing mean absolute SHAP values.

        Returns
        -------
        FeatureImportanceExplanation
            Object containing global feature-importance information (e.g., mean absolute SHAP values).
        """
        df_pandas = self._prepare_data()
        # Fit SHAP explainer
        explainer = self._get_explainer(df_pandas)
        shap_values = explainer(df_pandas[self.feature_cols])

        # Aggregate feature importances (mean absolute SHAP)
        mean_abs_shap = shap_values.abs.mean(0).values
        feature_importances = dict(zip(self.feature_cols, mean_abs_shap))

        # (Optional) Sort features by descending importance for the plot
        sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        sorted_features = [item[0] for item in sorted_importances]
        sorted_values = [item[1] for item in sorted_importances]

        # Create a bar plot using Matplotlib
        plt.figure()  # single distinct figure
        plt.bar(range(len(sorted_values)), sorted_values)
        plt.xticks(range(len(sorted_values)), sorted_features, rotation=45, ha="right")
        plt.title("Global Feature Importances (Mean Absolute SHAP)")
        plt.tight_layout()
        # Save as PNG
        plt.savefig("shap_bar_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Build a FeatureImportanceExplanation object
        explanation = FeatureImportanceExplanation(
            feature_importances=feature_importances,
            method="SHAP",
            description="Global feature importance via mean absolute SHAP values"
        )
        return explanation

    def explain_row(self, row: SparkDataFrame) -> FeatureImportanceExplanation:
        """
        Generates a SHAP explanation for a single row (or small subset).

        Parameters
        ----------
        row : SparkDataFrame
            A Spark DataFrame containing exactly one row (or a small subset of rows) to explain.

        Returns
        -------
        FeatureImportanceExplanation
            Object containing per-feature SHAP values for the provided row(s).
        """
        row_pandas = row.toPandas()
        explainer = self._get_explainer(row_pandas)
        shap_values = explainer(row_pandas[self.feature_cols])

        # If single row: shap_values.values -> shape (1, n_features)
        # For multiple rows: shape (n_rows, n_features)
        row_feature_importances = {}
        for idx, col in enumerate(self.feature_cols):
            # If multiple rows, you'd likely store arrays or handle them differently
            if shap_values.values.ndim == 2:
                row_feature_importances[col] = shap_values.values[0, idx]
            else:
                row_feature_importances[col] = shap_values.values[idx]

        explanation = FeatureImportanceExplanation(
            feature_importances=row_feature_importances,
            method="SHAP",
            description="Per-feature SHAP values for a single row"
        )
        return explanation

    # -------------------------------------------------------------------------
    # Internal/Helper Methods
    # -------------------------------------------------------------------------
    def _prepare_data(self) -> pd.DataFrame:
        """
        Converts the Spark DataFrame to a pandas DataFrame, sampling if necessary,
        and selects the appropriate feature columns.

        Returns
        -------
        pd.DataFrame
        """
        # Sample to avoid memory blow-ups in single-node
        df_spark_sampled = self.data.limit(self.max_samples)
        df_pandas = df_spark_sampled.toPandas()

        # If feature_cols not provided, use all columns except label_col
        if not self.feature_cols:
            all_cols = list(df_pandas.columns)
            if self.label_col and self.label_col in all_cols:
                all_cols.remove(self.label_col)
            self.feature_cols = all_cols

        return df_pandas

    def _get_explainer(self, df_pandas: pd.DataFrame) -> shap.Explainer:
        """
        Instantiates (or caches) the SHAP explainer object based on the model.

        Parameters
        ----------
        df_pandas : pd.DataFrame
            The local dataframe used for a background/reference distribution if needed.

        Returns
        -------
        shap.Explainer
        """
        # For tree-based models: shap.TreeExplainer(...)
        # For more general models: shap.Explainer(...)
        return shap.Explainer(
            self.model.predict,
            df_pandas[self.feature_cols],
            feature_names=self.feature_cols
        )
