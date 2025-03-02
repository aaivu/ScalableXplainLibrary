from scalableXplain.explainers.abstract_explainer import AbstractExplainer
from pyspark.sql import DataFrame

class DistributedExplainer(AbstractExplainer):
    def __init__(self, model, data: DataFrame):
        super().__init__(model)
        self.data = data
