class BaseExplainer:
    def __init__(self, model, data, task_type="classification"):
        self.model = model
        self.task_type = task_type
        self.data = data
        self.backend = self._detect_backend(data)

    def _detect_backend(self, data):
        if "pyspark.sql.dataframe.DataFrame" in str(type(data)):
            return "pyspark"
        elif "pandas.core.frame.DataFrame" in str(type(data)):
            return "pandas"
        else:
            raise ValueError("Unsupported data type")

    def explain(self, instance):
        raise NotImplementedError("Must implement in subclass")
