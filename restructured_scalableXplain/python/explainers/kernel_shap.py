import shap
from .base import BaseExplainer

class KernelSHAPExplainer(BaseExplainer):
    def explain(self, instance):
        if self.backend == "pandas":
            explainer = shap.KernelExplainer(self.model.predict, self.data)
            return explainer.shap_values(instance)
        elif self.backend == "pyspark":
            raise NotImplementedError("KernelSHAP for PySpark not supported directly.")
