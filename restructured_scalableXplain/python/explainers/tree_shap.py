from .base import BaseExplainer
import shap

class TreeSHAPExplainer(BaseExplainer):
    def explain(self, instance):
        if self.backend == "pandas":
            explainer = shap.TreeExplainer(self.model)
            return explainer.shap_values(instance)
        elif self.backend == "pyspark":
            instance_local = instance.toPandas()
            explainer = shap.TreeExplainer(self.model)
            return explainer.shap_values(instance_local)
