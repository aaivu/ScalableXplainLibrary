import numpy as np
import pandas as pd
import shap
from typing import Optional, Any
from pyspark.sql import DataFrame as SparkDataFrame
from scalableXplain.explainers.single_node.single_node_explainer import SingleNodeExplainer


class TreeSHAPSingleNodeExplainer(SingleNodeExplainer):
    """
    A single-node SHAP explainer for tree-based models using TreeSHAP.
    """

    def __init__(
        self,
        model: Any,
        data: SparkDataFrame,
        feature_cols: Optional[list[str]] = None,
        label_col: Optional[str] = None,
        max_samples: int = 1000,
    ):
        """
        Parameters
        ----------
        model : Any
            A tree-based model (e.g., RandomForest, XGBoost, LightGBM).
        data : SparkDataFrame
            The dataset in Spark format.
        feature_cols : list of str, optional
            Names of the feature columns.
        label_col : str, optional
            Name of the label column.
        max_samples : int, optional
            Maximum rows to sample from Spark DataFrame.
        """
        super().__init__(model=model, data=data)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.max_samples = max_samples

    def explain(self) -> dict:
        """
        Computes global feature importances using mean absolute TreeSHAP values.

        Returns
        -------
        dict
            Dictionary of feature importances `{feature_name: importance}`.
        """
        df_pandas = self._prepare_data()
        explainer = self._get_explainer(df_pandas)
        shap_values = explainer.shap_values(df_pandas[self.feature_cols])

        # Handle multi-class classification by averaging across classes
        if isinstance(shap_values, list):
            shap_array = np.mean(np.array(shap_values), axis=0)  # (samples, features)
        else:
            shap_array = shap_values  # Single-class case

        mean_abs_shap = np.abs(shap_array).mean(axis=0)  # Aggregate SHAP values
        feature_importances = dict(zip(self.feature_cols, mean_abs_shap))

        print("\n🔹 Global Feature Importances (TreeSHAP):")
        for feature, importance in feature_importances.items():
            # Convert array to float or list
            if isinstance(importance, np.ndarray):
                print(f"  {feature}: {importance.tolist()}")  # Multi-class case
            else:
                print(f"  {feature}: {float(importance):.6f}")  # Single-class case

        return feature_importances

    # def explain_row(self, row: SparkDataFrame) -> dict:
    #     """
    #     Computes SHAP values for a single row.
    #
    #     Parameters
    #     ----------
    #     row : SparkDataFrame
    #         A single-row Spark DataFrame.
    #
    #     Returns
    #     -------
    #     dict
    #         Dictionary of feature SHAP values `{feature_name: SHAP value}`.
    #     """
    #     row_pandas = row.toPandas()
    #     explainer = self._get_explainer(row_pandas)
    #     shap_values = explainer.shap_values(row_pandas[self.feature_cols])
    #
    #     # Handle multi-class output by averaging across classes
    #     if isinstance(shap_values, list):
    #         row_values = np.mean(np.array(shap_values), axis=0)  # (samples, features)
    #     else:
    #         row_values = shap_values  # Single-class case
    #
    #     row_feature_importances = dict(zip(self.feature_cols, row_values[0]))  # Extract first row
    #
    #     print("\n🔹 SHAP Values for Single Row:")
    #     for feature, shap_value in row_feature_importances.items():
    #         # Convert array to scalar or list
    #         if isinstance(shap_value, np.ndarray):
    #             print(f"  {feature}: {shap_value.tolist()}")  # Multi-class case
    #         else:
    #             print(f"  {feature}: {float(shap_value):.6f}")  # Single-class case
    #
    #     return row_feature_importances

    def _prepare_data(self) -> pd.DataFrame:
        """
        Converts the Spark DataFrame to pandas and selects features.

        Returns
        -------
        pd.DataFrame
        """
        df_spark_sampled = self.data.limit(self.max_samples)
        df_pandas = df_spark_sampled.toPandas()

        if not self.feature_cols:
            all_cols = list(df_pandas.columns)
            if self.label_col and self.label_col in all_cols:
                all_cols.remove(self.label_col)
            self.feature_cols = all_cols

        return df_pandas

    def _get_explainer(self, df_pandas: pd.DataFrame) -> shap.TreeExplainer:
        """
        Creates the SHAP TreeExplainer for the model.

        Returns
        -------
        shap.TreeExplainer
        """
        return shap.TreeExplainer(self.model)
