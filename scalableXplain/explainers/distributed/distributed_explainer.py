from abc import abstractmethod
from pyspark.sql import DataFrame as SparkDataFrame

# Import the higher-level abstract explainer interface:
from scalableXplain.explainers.abstract_explainer import AbstractExplainer

class DistributedExplainer(AbstractExplainer):
    """
    Abstract base class for single-node explainers.

    Inherits from `AbstractExplainer` to ensure we have at least `explain()` defined.
    This class adds specific references to a local model object and the Spark DataFrame
    containing the data to be explained.

    Concrete single-node explainers (e.g., SHAP, LIME) should subclass this and
    implement the `explain()` and optionally `explain_row()` methods.
    """

    def __init__(self, data: SparkDataFrame):
        """
        Parameters
        ----------
        model : object
            A trained model that can be used for generating predictions locally.
            For example, a scikit-learn model, or any model exposing .predict().
        data : SparkDataFrame
            The Spark DataFrame containing features (and possibly labels) to be explained.
        """
        super().__init__()  # Call the base AbstractExplainer constructor (if needed)
        self.data = data

    @abstractmethod
    def explain(self, *args, **kwargs):
        """
        Produce an explanation for the entire Spark dataset (or a subset).

        Concrete subclasses should implement the logic to:
          1) Possibly convert `self.data` to pandas,
          2) Run the explanation algorithm (e.g. SHAP, LIME),
          3) Return an explanation object (e.g., FeatureImportanceExplanation).

        Returns
        -------
        Explanation
            A subclass of Explanation (e.g. FeatureImportanceExplanation) containing
            results of the explanation.
        """
        raise NotImplementedError("Subclasses must implement the `explain` method.")

    def explain_row(self, row: SparkDataFrame, *args, **kwargs):
        """
        Optionally produce an explanation for a single row (or small subset).

        By default, it raises NotImplementedError. Concrete subclasses can override
        this if row-level explanations are supported.

        Parameters
        ----------
        row : SparkDataFrame
            A Spark DataFrame containing the row(s) to be explained.

        Returns
        -------
        Explanation
            The resulting explanation object for that row.
        """
        raise NotImplementedError("This single-node explainer does not support per-row explanations.")
