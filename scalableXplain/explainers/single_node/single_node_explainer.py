from ..abstract_explainer import AbstractExplainer

class SingleNodeExplainer(AbstractExplainer):
    """
    Base class for single-node explainers.
    """
    def __init__(self, model):
        self.model = model

    def explain(self, X):
        """
        Default explanation method.
        """
        raise NotImplementedError("Subclasses must implement the `explain` method.")

    def explain_row(self, row):
        """
        Explain a single row of data.
        """
        return self.explain([row])[0]
