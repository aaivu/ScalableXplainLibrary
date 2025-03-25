# import numpy as np
# import pandas as pd
# import shap
# from typing import Optional, Any
# from pyspark.sql import DataFrame as SparkDataFrame
# from scalableXplain.explainers.single_node.feature_importance.sn_feature_importance import SNFeatureImportancelainer
#
#
# class KernelSHAPSingleNodeExplainer(SNFeatureImportancelainer):
#     """
#     A single-node SHAP explainer using KernelSHAP.
#     """
#
#     def __init__(
#             self,
#             model: Any,
#             data: SparkDataFrame,
#             feature_cols: Optional[list[str]] = None,
#             label_col: Optional[str] = None,
#             max_samples: int = 1000,
#     ):
#         """
#         Parameters
#         ----------
#         model : Any
#             Any predictive model (e.g., neural networks, SVMs, tree-based models).
#         data : SparkDataFrame
#             The dataset in Spark format.
#         feature_cols : list of str, optional
#             Names of the feature columns.
#         label_col : str, optional
#             Name of the label column.
#         max_samples : int, optional
#             Maximum rows to sample from Spark DataFrame.
#         """
#         super().__init__(model=model, data=data)
#         self.feature_cols = feature_cols
#         self.label_col = label_col
#         self.max_samples = max_samples
#
#     def explain(self, num_samples: int = 100) -> dict:
#         """
#         Computes global feature importances using mean absolute KernelSHAP values.
#
#         Parameters
#         ----------
#         num_samples : int, optional
#             The number of samples for KernelSHAP estimation.
#
#         Returns
#         -------
#         dict
#             Dictionary of feature importances `{feature_name: importance}`.
#         """
#         df_pandas = self._prepare_data()
#         explainer = self._get_explainer(df_pandas)
#         shap_values = explainer.shap_values(df_pandas[self.feature_cols], nsamples=num_samples)
#
#         mean_abs_shap = np.abs(shap_values).mean(axis=0)  # Aggregate SHAP values
#         feature_importances = dict(zip(self.feature_cols, mean_abs_shap))
#
#         print("\n🔹 Global Feature Importances (KernelSHAP):")
#         for feature, importance in feature_importances.items():
#             print(f"  {feature}: {float(importance):.6f}")
#
#         return feature_importances
#
#     def explain_row(self, row: SparkDataFrame, num_samples: int = 100) -> dict:
#         """
#         Computes SHAP values for a single row.
#
#         Parameters
#         ----------
#         row : SparkDataFrame
#             A single-row Spark DataFrame.
#         num_samples : int, optional
#             The number of samples for KernelSHAP estimation.
#
#         Returns
#         -------
#         dict
#             Dictionary of feature SHAP values `{feature_name: SHAP value}`.
#         """
#         row_pandas = row.toPandas()
#         explainer = self._get_explainer(row_pandas)
#         shap_values = explainer.shap_values(row_pandas[self.feature_cols], nsamples=num_samples)
#
#         row_feature_importances = dict(zip(self.feature_cols, shap_values[0]))
#
#         print("\n🔹 SHAP Values for Single Row:")
#         for feature, shap_value in row_feature_importances.items():
#             print(f"  {feature}: {float(shap_value):.6f}")
#
#         return row_feature_importances
#
#     def _prepare_data(self) -> pd.DataFrame:
#         """
#         Converts the Spark DataFrame to pandas and selects features.
#
#         Returns
#         -------
#         pd.DataFrame
#         """
#         df_spark_sampled = self.data.limit(self.max_samples)
#         df_pandas = df_spark_sampled.toPandas()
#
#         if not self.feature_cols:
#             all_cols = list(df_pandas.columns)
#             if self.label_col and self.label_col in all_cols:
#                 all_cols.remove(self.label_col)
#             self.feature_cols = all_cols
#
#         return df_pandas
#
#     def _get_explainer(self, df_pandas: pd.DataFrame) -> shap.KernelExplainer:
#         """
#         Creates the SHAP KernelExplainer for the model.
#
#         Returns
#         -------
#         shap.KernelExplaine
#         """
#         return shap.KernelExplainer(self.model.predict, df_pandas[self.feature_cols])