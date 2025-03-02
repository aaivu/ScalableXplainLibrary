# explanations/feature_importance_explanation.py

class FeatureImportanceExplanation:
    """
    A simple container to store feature importance results.
    """
    def __init__(self,
                 feature_importances: dict[str, float],
                 method: str,
                 description: str):
        """
        Parameters
        ----------
        feature_importances : dict
            A dictionary of {feature_name: importance_score}.
        method : str
            Identifier for the explanation method (e.g. "SHAP", "LIME").
        description : str
            Human-readable description of how these importances were derived.
        """
        self.feature_importances = feature_importances
        self.method = method
        self.description = description

    def __repr__(self):
        return (f"FeatureImportanceExplanation(method={self.method}, "
                f"importances={self.feature_importances})")
