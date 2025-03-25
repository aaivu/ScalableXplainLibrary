# import numpy as np
# import pandas as pd
# import shap
# from typing import Optional, Any
# from pyspark.sql import DataFrame as SparkDataFrame
# from scalableXplain.explainers.single_node.feature_importance.sn_feature_importance import SNFeatureImportancelainer
#
#
# class TreeSHAPSingleNodeExplainer(SNFeatureImportancelainer):
#     """
#     A single-node SHAP explainer for tree-based models using TreeSHAP.
#     """
#     def __init__(
#         self,
#         model: Any,
#         data: SparkDataFrame,
#         feature_cols: Optional[list[str]] = None,
#         label_col: Optional[str] = None,
#         max_samples: int = 1000,
#     ):
#         """
#         Parameters
#         ----------
#         model : Any
#             A tree-based model (e.g., RandomForest, XGBoost, LightGBM).
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
#     def explain(self) -> dict:
#         """
#         Computes global feature importances using mean absolute TreeSHAP values.
#
#         Returns
#         -------
#         dict
#             Dictionary of feature importances `{feature_name: importance}`.
#         """
#         df_pandas = self._prepare_data()
#         explainer = self._get_explainer(df_pandas)
#         shap_values = explainer.shap_values(df_pandas[self.feature_cols])
#
#         # Handle multi-class classification by averaging across classes
#         if isinstance(shap_values, list):
#             shap_array = np.mean(np.array(shap_values), axis=0)  # (samples, features)
#         else:
#             shap_array = shap_values  # Single-class case
#
#         mean_abs_shap = np.abs(shap_array).mean(axis=0)  # Aggregate SHAP values
#         feature_importances = dict(zip(self.feature_cols, mean_abs_shap))
#
#         print("\n🔹 Global Feature Importances (TreeSHAP):")
#         for feature, importance in feature_importances.items():
#             # Convert array to float or list
#             if isinstance(importance, np.ndarray):
#                 print(f"  {feature}: {importance.tolist()}")  # Multi-class case
#             else:
#                 print(f"  {feature}: {float(importance):.6f}")  # Single-class case
#
#         return feature_importances
#
#     def explain_row(self, row: SparkDataFrame) -> dict:
#         """
#         Computes SHAP values for a single row.
#
#         Parameters
#         ----------
#         row : SparkDataFrame
#             A single-row Spark DataFrame.
#
#         Returns
#         -------
#         dict
#             Dictionary of feature SHAP values `{feature_name: SHAP value}`.
#         """
#         row_pandas = row.toPandas()
#         explainer = self._get_explainer(row_pandas)
#         shap_values = explainer.shap_values(row_pandas[self.feature_cols])
#
#         # Handle multi-class output by averaging across classes
#         if isinstance(shap_values, list):
#             row_values = np.mean(np.array(shap_values), axis=0)  # (samples, features)
#         else:
#             row_values = shap_values  # Single-class case
#
#         row_feature_importances = dict(zip(self.feature_cols, row_values))  # Extract first row
#
#         print("\n🔹 SHAP Values for Single Row:")
#         for feature, shap_value in row_feature_importances.items():
#             # Convert array to scalar or list
#             if isinstance(shap_value, np.ndarray):
#                 print(f"  {feature}: {shap_value.tolist()}")  # Multi-class case
#             else:
#                 print(f"  {feature}: {float(shap_value):.6f}")  # Single-class case
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
#     def _get_explainer(self, df_pandas: pd.DataFrame) -> shap.TreeExplainer:
#         """
#         Creates the SHAP TreeExplainer for the model.
#
#         Returns
#         -------
#         shap.TreeExplainer
#         """
#         return shap.TreeExplainer(self.model)

import tempfile
import shutil
import numpy as np
import pandas as pd
import shap
from typing import Optional, Any
from pyspark.sql import DataFrame as SparkDataFrame
from scalableXplain.explainers.single_node.feature_importance.sn_feature_importance import SNFeatureImportancelainer
from pyspark.sql.types import StructType, StructField, FloatType
from typing import Iterator
from pyspark.sql.functions import abs, col, mean
import os
import unittest
import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline

from scalableXplain.explainers.single_node.feature_importance.sn_feature_importance import SNFeatureImportancelainer
from pyspark.sql.types import StructType, StructField, FloatType
from typing import Iterator, Optional, Any
import pandas as pd
import shap


import unittest
import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import broadcast
from pyspark.ml.util import MLWritable, MLReader  # Import MLWritable, MLReader

from scalableXplain.explainers.single_node.feature_importance.sn_feature_importance import SNFeatureImportancelainer
from pyspark.sql.types import StructType, StructField, FloatType
from typing import Iterator, Optional, Any
import pandas as pd
import shap
import io
import pickle


import unittest
import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import broadcast
from pyspark.ml.util import MLWritable, MLReader  # Import MLWritable, MLReader
import tempfile
import os

from scalableXplain.explainers.single_node.feature_importance.sn_feature_importance import SNFeatureImportancelainer
from pyspark.sql.types import StructType, StructField, FloatType
from typing import Iterator, Optional, Any
import pandas as pd
import shap
import io
import pickle


import unittest
import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import broadcast
from pyspark.ml.util import MLWritable, MLReader  # Import MLWritable, MLReader
import tempfile
import os
import errno

from scalableXplain.explainers.single_node.feature_importance.sn_feature_importance import SNFeatureImportancelainer
from pyspark.sql.types import StructType, StructField, FloatType
from typing import Iterator, Optional, Any
import pandas as pd
import shap
import io
import pickle


class TreeSHAPSingleNodeExplainer(SNFeatureImportancelainer):
    def __init__(
        self,
        model: Any,
        data: SparkDataFrame,
        feature_cols: Optional[list[str]] = None,
        label_col: Optional[str] = None,
        max_samples: int = 1000,
    ):
        super().__init__(model=model, data=data)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.max_samples = max_samples
        self.spark_context = data.sparkSession.sparkContext  # Get SparkContext from the DataFrame

        # Use a robust method to serialize the model
        self.broadcast_model = self._broadcast_model(model)

    def _broadcast_model(self, model):
        """
        Handles broadcasting the model, ensuring it's serializable.
        For PySpark models, we save and load to avoid pickling issues.
        """
        if isinstance(model, MLWritable):
            # Save the model to a temporary file
            temp_file_descriptor, temp_file_path = tempfile.mkstemp()  # Create a temporary file, get fd and name
            try:
                model.write().overwrite().save(temp_file_path)  # Save to the temporary file with overwrite
                if not os.path.isfile(temp_file_path):
                    raise IOError(f"Expected file at {temp_file_path} but found directory or nothing.")
                with open(temp_file_path, 'rb') as f:
                    model_bytes = f.read()
            finally:
                os.close(temp_file_descriptor)  # Close the file descriptor
                try:
                    os.unlink(temp_file_path)  # Manually remove the file.
                except OSError as e:
                    if e.errno != errno.ENOENT:  # Ignore "No such file or directory"
                        raise  # Re-raise other errors
            return self.spark_context.broadcast(model_bytes)
        else:
            # If it's not a PySpark model, try to pickle it directly
            try:
                pickled_model = pickle.dumps(model)
                return self.spark_context.broadcast(pickled_model)
            except Exception as e:
                raise ValueError(
                    "Model is not a PySpark MLWritable model and could not be pickled.  Ensure your model is serializable."
                ) from e

    def _load_model(self, model_bytes):
        """Load the model from bytes."""
        # Use BytesIO to read from the byte array
        with io.BytesIO(model_bytes) as buffer:
            #  Use MLReader.load
            loaded_model = Pipeline.load(buffer.name) # any pipeline will do, we just need the reader.
            return loaded_model.stages[-1]  # return the actual model.

    def explain(self) -> dict:
        """
        Compute SHAP values for the entire (sampled) Spark DataFrame in a distributed way,
        aggregate them, and return a dictionary of mean absolute SHAP values by feature.
        """
        # Sample the Spark DataFrame to avoid excessive computations if desired
        df_spark_sampled = self.data.limit(self.max_samples)

        # If feature_cols not set, determine them
        df_pandas_sampled = df_spark_sampled.toPandas()
        if not self.feature_cols:
            all_cols = list(df_pandas_sampled.columns)
            if self.label_col and self.label_col in all_cols:
                all_cols.remove(self.label_col)
            self.feature_cols = all_cols

        # Define our mapInPandas UDF to compute SHAP values on each partition
        def calculate_shap(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
            #  Get the model inside the worker
            #  Important:  Use the broadcasted model here.
            model_bytes = self.broadcast_model.value  # Access the broadcasted value
            local_model = self._load_model(model_bytes)
            local_feature_cols = self.feature_cols

            for pdf in iterator:
                data_for_shap = pdf[local_feature_cols]
                explainer = shap.TreeExplainer(local_model)  # Create explainer *inside* the worker
                shap_values = explainer.shap_values(
                    data_for_shap, check_additivity=False
                )

                # Handle classifiers that return a list of arrays (one per class)
                if isinstance(shap_values, list):
                    # By convention, pick the SHAP array for the first class
                    shap_values_first_class = shap_values[0]
                else:
                    shap_values_first_class = shap_values

                # Some tree models return 3D SHAP arrays [samples, base/feature, classes]
                # Typically you just want the first (or single) class
                if shap_values_first_class.ndim == 3:
                    shap_values_first_class = shap_values_first_class[:, :, 0]

                # Return SHAP values as pandas DataFrame with columns = feature names
                yield pd.DataFrame(shap_values_first_class, columns=local_feature_cols)

        # Spark schema for the returned SHAP DataFrame
        return_schema = StructType([StructField(f, FloatType()) for f in self.feature_cols])

        # Apply distributed SHAP calculation
        shap_values_sdf = df_spark_sampled.mapInPandas(calculate_shap, schema=return_schema)

        # Collect final SHAP values into Pandas to compute aggregates
        shap_values_pdf = shap_values_sdf.toPandas()

        # Aggregate (for example, mean absolute shap across samples)
        mean_abs_shap = shap_values_pdf.abs().mean(axis=0)
        result = mean_abs_shap.to_dict()

        return result

    def explain_row(self, row: SparkDataFrame) -> dict:
        """
        Compute SHAP values for a single Spark DataFrame row (or a small batch).
        Converts to pandas locally, runs shap_values, and returns the result as a dict.
        """
        # Convert to Pandas (should be just 1 row or a small batch)
        row_pdf = row.toPandas()
        # If feature_cols not set, figure it out
        if not self.feature_cols:
            all_cols = list(row_pdf.columns)
            if self.label_col and self.label_col in all_cols:
                all_cols.remove(self.label_col)
            self.feature_cols = all_cols

        # Local SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        data_for_shap = row_pdf[self.feature_cols]

        shap_values = explainer.shap_values(data_for_shap, check_additivity=False)

        # Handle multi-class or 3D returns
        if isinstance(shap_values, list):
            shap_values_first_class = shap_values[0]
        else:
            shap_values_first_class = shap_values

        if shap_values_first_class.ndim == 3:
            shap_values_first_class = shap_values_first_class[:, :, 0]

        # If row_pdf is just one row, shap_values_first_class should have shape (1, #features)
        # Convert to dict
        shap_result = {}
        for i, feat in enumerate(self.feature_cols):
            shap_result[feat] = float(shap_values_first_class[0, i])

        return shap_result

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
