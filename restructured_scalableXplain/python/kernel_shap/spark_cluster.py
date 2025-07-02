# kernel_shap/spark_cluster.py

from synapse.ml.explainers import TabularSHAP
from pyspark.sql.functions import rand, broadcast
from core.explanations import FeatureImportanceExplanation

class SparkKernelSHAPExplainer(FeatureImportanceExplanation):
    def __init__(self, model, input_cols, target_col="probability", target_classes=[1], background_data=None, num_samples=5000):
        """
        model: Trained PySpark ML pipeline model
        input_cols: List of feature column names before vector assembler
        target_col: Name of the model output column to explain
        target_classes: Class indices to explain (for classification)
        background_data: Optional. If None, will sample from training data later
        num_samples: Number of SHAP samples
        """
        self.model = model
        self.input_cols = input_cols
        self.target_col = target_col
        self.target_classes = target_classes
        self.num_samples = num_samples
        self.background_data = background_data
        self.explainer = None

    def build_explainer(self, training_data):
        """
        Prepares the TabularSHAP explainer using the training dataset.
        """
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
        """
        Returns a DataFrame with added 'shapValues' column.
        Requires that `build_explainer()` has been called.
        """
        if self.explainer is None:
            raise ValueError("Explainer has not been built. Call build_explainer() first.")
        return self.explainer.transform(instances_df)
