from scalableXplain.explainers.single_node.single_node_explainer import SingleNodeExplainer
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from sklearn.utils import check_array


class SingleNodeLIME(SingleNodeExplainer):
    """
    LIME implementation for single-node classification models.
    """

    def __init__(self, model, data, feature_names=None, categorical_features=None,
                 categorical_names=None, kernel_width=None, discretize_continuous=True):
        """
        :param model: Trained classification model with .predict_proba(X)
        :param data: Background data for training the LIME explainer
        :param feature_names: List of feature names (optional)
        :param categorical_features: List of indices for categorical features (optional)
        :param categorical_names: Dict mapping categorical feature indices to names (optional)
        :param kernel_width: Bandwidth for the LIME kernel. If None, defaults to sqrt(num_features)*0.75
        :param discretize_continuous: Whether to discretize continuous features
        """
        super().__init__(model)

        # Convert background data to a numeric NumPy array
        self.data = check_array(data, ensure_2d=True, dtype=float)

        # LIME requires a predict_proba function for classification tasks
        def predict_fn(X):
            return self.model.predict_proba(X)

        self.predict_fn = predict_fn

        # Initialize the LimeTabularExplainer
        self.explainer = LimeTabularExplainer(
            training_data=self.data,
            feature_names=feature_names,
            categorical_features=categorical_features,
            categorical_names=categorical_names,
            kernel_width=kernel_width,
            discretize_continuous=discretize_continuous
        )

    def explain(self, X, num_features=5, labels=(0,)):
        """
        Compute LIME explanations for each sample in X.
        :param X: Samples for which to generate local explanations
        :param num_features: Maximum number of features to include in explanation
        :param labels: Tuple or list of class indices for which to generate explanations
        :return: A list of LIME explanation objects
        """
        X = check_array(X, ensure_2d=True, dtype=float)

        explanations = []
        for row in X:
            exp = self.explainer.explain_instance(
                data_row=row,
                predict_fn=self.predict_fn,
                num_features=num_features,
                labels=labels
            )
            explanations.append(exp)
        return explanations

    def explain_row(self, row, num_features=5, labels=(0,)):
        """
        Explain a single row using LIME.
        :param row: 1D array representing a single sample
        :param num_features: Maximum number of features to include in explanation
        :param labels: Tuple or list of class indices for which to generate explanations
        :return: LIME explanation object
        """
        # Wrap row to form a 2D array
        row_2d = check_array([row], ensure_2d=True, dtype=float)

        # Since "explain" already handles multiple rows, we can just reuse it for a single row
        explanation = self.explain(row_2d, num_features=num_features, labels=labels)

        # explanation is a list (one explanation per row); we only have one row
        return explanation[0]
