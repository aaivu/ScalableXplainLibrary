from .abstract_explanation import AbstractExplanation

class FeatureImportanceExplanation(AbstractExplanation):
    """
    Stores feature importance values from SHAP.
    """

    def __init__(self, feature_importances, method="SHAP"):
        self.feature_importances = feature_importances
        self.method = method

    def to_dict(self):
        return {
            "method": self.method,
            "feature_importances": self.feature_importances
        }

    def plot(self):
        """Plot using a bar plot."""
        from scalableXplain.plots.bar_plot import BarPlot
        plot = BarPlot(self.feature_importances)
        plot.render()
