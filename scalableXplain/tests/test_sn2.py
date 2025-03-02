import shap
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, DoubleType

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("KernelSHAP_Example") \
    .getOrCreate()

# Step 2: Create Sample Dataset
data = spark.createDataFrame([
    (1, 5.1, 3.5, 1.4, 0.2),
    (0, 4.9, 3.0, 1.4, 0.2),
    (1, 6.2, 3.4, 5.4, 2.3),
    (0, 5.9, 3.0, 5.1, 1.8),
], ["label", "feature1", "feature2", "feature3", "feature4"])

# Step 3: Prepare Features
feature_cols = ["feature1", "feature2", "feature3", "feature4"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

# Step 4: Train a Machine Learning Model
lr = LogisticRegression(featuresCol="features", labelCol="label")
model = lr.fit(data)

# Step 5: Convert Data to Pandas (for SHAP Kernel Explainer)
pandas_data = data.select("features").toPandas()
X = np.array(pandas_data["features"].tolist())


def predict_fn(X):
    # Convert NumPy array to Pandas DataFrame
    pandas_df = pd.DataFrame(X, columns=feature_cols)

    # Convert Pandas DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(pandas_df)

    # Assemble features into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    spark_df = assembler.transform(spark_df)

    # Perform prediction using Spark model
    predictions = model.transform(spark_df).select("probability").toPandas()

    # Extract probabilities for the positive class
    return np.array([p[1] for p in predictions["probability"]])


# Initialize SHAP Kernel Explainer
explainer = shap.KernelExplainer(predict_fn, X)
shap_values = explainer.shap_values(X)

# Step 7: Convert SHAP Values to PySpark DataFrame
shap_df = pd.DataFrame(shap_values, columns=feature_cols)
shap_spark_df = spark.createDataFrame(shap_df)

# Display SHAP Values
shap_spark_df.show()
