# core/explanations.py
from abc import ABC, abstractmethod

class FeatureImportanceExplainer(ABC):
    @abstractmethod
    def explain_instance(self, instance):
        pass

    @abstractmethod
    def explain_dataframe(self, df):
        pass
