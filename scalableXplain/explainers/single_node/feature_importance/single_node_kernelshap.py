from scalableXplain.explainers.single_node.single_node_explainer import SingleNodeExplainer
import shap
import numpy as np
from sklearn.utils import check_array

class SingleNodeKernelSHAP(SingleNodeExplainer):
    """
    KernelSHAP implementation for single-node classification models.
    """

    def __init__(self, model, data):
        super().__init__(model)
        # Convert background data to a numeric NumPy array
        self.data = check_array(data, ensure_2d=True, dtype=float)

        # For classification models, use predict_proba so that SHAP gets a numeric output
        def predict_fn(X):
            return self.model.predict_proba(X)

        self.explainer = shap.KernelExplainer(predict_fn, self.data)

    def explain(self, X):
        """
        Compute KernelSHAP explanations for given input X.
        Returns a NumPy array or list of arrays (multiclass).
        """
        X = check_array(X, ensure_2d=True, dtype=float)
        shap_values = self.explainer.shap_values(X)
        return shap_values

    def explain_row(self, row):
        """
        Explain a single row using KernelSHAP.
        Returns the SHAP values for that row.
        """
        # Wrap row to form a 2D array
        row_2d = check_array([row], ensure_2d=True, dtype=float)
        shap_values = self.explain(row_2d)

        # For multiclass, shap_values can be a list of arrays: one per class
        # If you want the explanation for each class, return shap_values as is
        # If you only want a single class's explanation, pick an index
        return shap_values
