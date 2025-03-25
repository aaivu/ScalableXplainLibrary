from scalableXplain.explainers.single_node.single_node_explainer import SingleNodeExplainer
import shap
from sklearn.utils import check_array

class SingleNodeTreeSHAP(SingleNodeExplainer):
    """
    TreeSHAP implementation for single-node tree-based models.
    """

    def __init__(self, model):
        if not hasattr(model, "tree_"):
            raise ValueError("Model must be a tree-based model with a `tree_` attribute.")
        super().__init__(model)
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, X):
        """
        Compute TreeSHAP explanations for given input X.
        """
        X = check_array(X)
        shap_values = self.explainer.shap_values(X)
        return shap_values

    def explain_row(self, row):
        """
        Explain a single row using TreeSHAP.
        """
        return self.explain([row])[0]
