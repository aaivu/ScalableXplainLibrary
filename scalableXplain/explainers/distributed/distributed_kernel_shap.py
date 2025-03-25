from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql import functions as F
from pyspark.ml.linalg import DenseVector
from synapse.ml.explainers import TabularSHAP
from scalableXplain.explainers.distributed.distributed_explainer import DistributedExplainer


class DistributedTabularSHAP(DistributedExplainer):
    """
    A distributed TabularSHAP explainer using SynapseML for Spark DataFrames.
    Inherits from DistributedExplainer (itself from AbstractExplainer).
    """

    def __init__(
        self,
        model,
        data: SparkDataFrame,
        input_cols,
        target_col="probability",
        target_classes=[1],
        output_col="shapValues",
        num_samples=5000,
        background_data=None,
    ):
        """
        Parameters
        ----------
        model : pyspark.ml.Model
            A trained Spark ML model (LogisticRegression, DecisionTree, etc.).
        data : SparkDataFrame
            Spark DataFrame containing the raw features (not vectorized) for TabularSHAP.
        input_cols : list of str
            Feature columns to be explained.
        target_col : str
            Column of the model output to explain (often 'probability').
        target_classes : list of int
            Class indices to explain. [1] = the positive class in binary classification.
        output_col : str
            Column name in which the SHAP values are stored.
        num_samples : int
            Number of samples used by KernelSHAP inside TabularSHAP.
        background_data : SparkDataFrame
            A small subset of rows used for “background” in SHAP. If None, we pick 10% of the data.
        """
        super().__init__(data=data)
        self.model = model
        self.input_cols = input_cols
        self.target_col = target_col
        self.target_classes = target_classes
        self.output_col = output_col
        self.num_samples = num_samples

        # If user doesn't specify background, pick 10% of the data
        if background_data is None:
            total_rows = self.data.count()
            ten_percent_rows = int(total_rows * 0.1)
            background_data = self.data.orderBy(F.rand()).limit(ten_percent_rows)

        self.background_data = background_data



    def explain(self, data=None):
        """
        Compute TabularSHAP for the entire dataset (or subset) and return a dictionary with
        the mean absolute SHAP value for every feature in input_cols.
        """
        if data is None:
            data = self.data

        shap = TabularSHAP(
            inputCols=self.input_cols,
            outputCol=self.output_col,
            numSamples=self.num_samples,
            model=self.model,
            targetCol=self.target_col,
            targetClasses=self.target_classes,
            backgroundData=self.background_data,
        )
        shap_df = shap.transform(data)

        # Define a UDF that converts the output to a flat list of floats.
        def vector_to_array_udf(v):
            if v is None:
                return None
            try:
                arr = v.toArray().tolist()
            except AttributeError:
                arr = list(v)
            flat_list = []
            for item in arr:
                if isinstance(item, DenseVector):
                    flat_list.extend(item.toArray().tolist())
                else:
                    flat_list.append(float(item))
            return flat_list

        vector_to_array = F.udf(vector_to_array_udf, ArrayType(DoubleType()))

        # Convert the SHAP output to an array column.
        shap_df = shap_df.withColumn("shap_array", vector_to_array(F.col(self.output_col)))

        # Create aggregation expressions to compute mean absolute SHAP value for each feature.
        agg_exprs = [
            F.avg(F.abs(F.col("shap_array")[i])).alias(feature)
            for i, feature in enumerate(self.input_cols)
        ]

        # Perform the aggregation.
        global_shap_df = shap_df.agg(*agg_exprs)

        # Collect the result as a dictionary.
        result = global_shap_df.collect()[0].asDict()
        return result

    def explain_row(self, row: SparkDataFrame):
        """
        Compute TabularSHAP for a single-row Spark DataFrame subset.
        Returns a dictionary with global SHAP values for each feature.
        """
        return self.explain(data=row)
