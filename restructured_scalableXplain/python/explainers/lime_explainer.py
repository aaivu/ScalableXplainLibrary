from .base import BaseExplainer
from lime.lime_tabular import LimeTabularExplainer

class LIMEExplainer(BaseExplainer):
    def explain(self, instance):
        if self.backend == "pandas":
            explainer = LimeTabularExplainer(
                training_data=self.data.values,
                feature_names=self.data.columns.tolist(),
                mode=self.task_type
            )
            return explainer.explain_instance(instance.values[0], self.model.predict_proba)
        elif self.backend == "pyspark":
            raise NotImplementedError("LIME is not natively supported for PySpark.")
