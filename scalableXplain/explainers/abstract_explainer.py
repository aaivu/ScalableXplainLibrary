"""
Abstract base class for all types of explainers.

This module defines a universal Explainer interface, enforced via Python's
Abstract Base Class (ABC). Subclasses (e.g. single-node or distributed)
should implement the required methods to provide explanations.
"""

from abc import ABC, abstractmethod

class AbstractExplainer(ABC):
    """
    Abstract base class for any explainer in this package.

    This interface is intentionally broad. Specific implentations
    (like SingleNodeExplainer or DistributedExplainer) can refine or extend
    it to suit their environment (e.g., local vs. Spark vs. other distributed frameworks).
    """

    @abstractmethod
    def explain(self, *args, **kwargs):
        """
        Produce an explanation for a dataset or a subset of data.

        Returns
        -------
        Explanation
            The resulting explanation object. The exact type of Explanation
            may depend on the concrete implementation (FeatureImportanceExplanation,
            ThresholdTreeExplanation, etc.).
        """
        raise NotImplementedError("Subclasses must implement the `explain` method.")

    # def explain_row(self, *args, **kwargs):
    #     """
    #     Optionally produce an explanation for a single row (or small subset of rows).
    #
    #     By default, it raises NotImplementedError. Concrete subclasses can override
    #     this if row-level explanations are supported.
    #
    #     Parameters
    #     ----------
    #     *args, **kwargs :
    #         Flexible signature to accommodate various data formats (Spark rows,
    #         pandas DataFrames, NumPy arrays, etc.).
    #
    #     Returns
    #     -------
    #     Explanation
    #         A specialized explanation object for the given row(s).
    #     """
    #     raise NotImplementedError("This explainer does not support per-row explanations.")
