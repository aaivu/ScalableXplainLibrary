# kernel_shap/single_node.py

from shap import KernelExplainer
from core.explanations import FeatureImportanceExplainer

class KernelSHAPExplainer(FeatureImportanceExplainer):
    def __init__(self, model, background_data, feature_names=None):
        self.feature_names = feature_names
        self.explainer = KernelExplainer(model.predict, background_data, feature_names)

    def explain_instance(self, instance):
        shap_values = self.explainer.shap_values(instance)
        return shap_values

    def explain_dataframe(self, df):
        shap_values = self.explainer.shap_values(df)
        return shap_values
