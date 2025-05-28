
import shap
from .base import BaseExplainer
import pandas as pd

class TreeSHAPExplainer(BaseExplainer):
    def __init__(self, model, background_data):
        super().__init__(model, background_data)
        if self.backend == "pandas":
            self.explainer = shap.TreeExplainer(model)
        else:
            raise NotImplementedError("TreeSHAP for PySpark is not supported yet.")

    def explain_row(self, instance):
        if self.backend == "pandas":
            shap_values = self.explainer.shap_values(instance)
            return shap_values
        else:
            raise NotImplementedError("Row-wise TreeSHAP explanation not supported for Spark.")

    def explain(self, df):
        if self.backend == "pandas":
            shap_values = self.explainer.shap_values(df)
            return shap_values
        else:
            raise NotImplementedError("Batch TreeSHAP explanation not supported for Spark.")
