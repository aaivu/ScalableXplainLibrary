

from abc import ABC, abstractmethod

class BaseExplainer(ABC):
    def __init__(self, model, background_data):
        self.model = model
        self.background_data = background_data
        self.backend = self._detect_backend()

    def _detect_backend(self):
        try:
            import pyspark
            from pyspark.sql import DataFrame as SparkDataFrame
            if isinstance(self.background_data, SparkDataFrame):
                return "spark"
        except ImportError:
            pass
        return "pandas"

    @abstractmethod
    def explain_row(self, instance):
        """
        Explain a single row/instance.
        """
        pass

    @abstractmethod
    def explain(self, df):
        """
        Explain a full DataFrame.
        """
        pass
