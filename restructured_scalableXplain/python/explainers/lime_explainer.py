
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
from .base import BaseExplainer

class LIMEExplainer(BaseExplainer):
    def __init__(self, model, background_data):
        super().__init__(model, background_data)
        if self.backend == "pandas":
            self.explainer = LimeTabularExplainer(
                training_data=np.array(background_data),
                feature_names=background_data.columns.tolist(),
                mode="classification" if hasattr(model, "predict_proba") else "regression"
            )
        else:
            raise NotImplementedError("LIME for PySpark is not supported yet.")

    def explain_row(self, instance):
        if self.backend == "pandas":
            explanation = self.explainer.explain_instance(
                instance.values[0],
                self.model.predict_proba if hasattr(self.model, "predict_proba") else self.model.predict
            )
            return explanation
        else:
            raise NotImplementedError("Row-wise LIME explanation not supported for Spark.")

    def explain(self, df):
        if self.backend == "pandas":
            return [self.explain_row(df.iloc[[i]]) for i in range(df.shape[0])]
        else:
            raise NotImplementedError("Batch LIME explanation not supported for Spark.")
