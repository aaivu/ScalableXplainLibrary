from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql import functions as F
from pyspark.ml.linalg import DenseVector
from synapse.ml.explainers import TabularLIME
from scalableXplain.explainers.distributed.distributed_explainer import DistributedExplainer


class DistributedTabularLIME(DistributedExplainer):
    """
    A distributed TabularLIME explainer using SynapseML for Spark DataFrames.
    Inherits from DistributedExplainer (itself from AbstractExplainer).

    This is structurally similar to DistributedTabularSHAP, but uses LIME as the
    underlying local explainer. The global explanation is computed as the mean
    absolute value of the local LIME coefficients for each feature.
    """

    def __init__(
            self,
            model,
            data: SparkDataFrame,
            input_cols,
            target_col="probability",
            target_classes=[1],
            output_col="limeValues",
            num_samples=2000,
            background_data=None,
    ):
        """
        Parameters
        ----------
        model : pyspark.ml.Model
            A trained Spark ML model (e.g. LogisticRegression, DecisionTree, etc.).
        data : SparkDataFrame
            Spark DataFrame containing the raw features (non-vectorized) for TabularLIME.
        input_cols : list of str
            Feature columns to be explained.
        target_col : str
            Column of the model output to explain (often 'probability').
        target_classes : list of int
            Class indices to explain. [1] = the positive class in binary classification, etc.
        output_col : str
            Column name in which the LIME values are stored.
        num_samples : int
            Number of samples used by LIME’s internal sampling approach.
        background_data : SparkDataFrame
            A small subset of rows used for “background” in LIME. If None, picks 10% of data.
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
        Compute TabularLIME for the entire dataset (or the provided subset),
        and return a dictionary with the mean absolute LIME value for each feature
        in self.input_cols. This is a very rough 'global' explanation measure.
        """
        if data is None:
            data = self.data

        # Build and run the LIME explainer
        lime = (
            TabularLIME()
            .setModel(self.model)
            .setInputCols(self.input_cols)
            .setTargetCol(self.target_col)
            .setTargetClasses(self.target_classes)
            .setNumSamples(self.num_samples)
            .setBackgroundData(self.background_data)
            .setOutputCol(self.output_col)
        )

        lime_df = lime.transform(data)

        # A helper UDF to convert vector columns into arrays of floats
        def vector_to_array_udf(v):
            if v is None:
                return None
            try:
                arr = v.toArray().tolist()
            except AttributeError:
                # If it's already a python list, or another structure
                arr = list(v)
            flat_list = []
            for item in arr:
                if isinstance(item, DenseVector):
                    flat_list.extend(item.toArray().tolist())
                else:
                    flat_list.append(float(item))
            return flat_list

        vector_to_array = F.udf(vector_to_array_udf, ArrayType(DoubleType()))

        # Convert the LIME output to an array column.
        lime_df = lime_df.withColumn("lime_array", vector_to_array(F.col(self.output_col)))

        # Compute the average absolute LIME value per feature across all rows
        agg_exprs = [
            F.avg(F.abs(F.col("lime_array")[i])).alias(feature)
            for i, feature in enumerate(self.input_cols)
        ]
        global_lime_df = lime_df.agg(*agg_exprs)

        # Collect as a dictionary (feature -> mean absolute LIME)
        result = global_lime_df.collect()[0].asDict()
        return result

    def explain_row(self, data):
        """
        Compute TabularLIME for a single-row Spark DataFrame (or a small subset).
        Returns a Spark DataFrame with local LIME explanations (i.e. per-row).
        """
        lime = (
            TabularLIME()
            .setModel(self.model)
            .setInputCols(self.input_cols)
            .setTargetCol(self.target_col)
            .setTargetClasses(self.target_classes)
            .setNumSamples(self.num_samples)
            .setBackgroundData(self.background_data)
            .setOutputCol(self.output_col)
        )

        lime_df = lime.transform(data)

        return lime_df
