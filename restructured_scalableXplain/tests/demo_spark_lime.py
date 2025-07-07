#!/usr/bin/env python3
"""
spark_lime_demo.py
-------------------------------------------------
Trains a logistic-regression model on the
Breast-Cancer-Wisconsin dataset with PySpark and
explains it using SynapseML’s Tabular LIME through
the SparkLIMEExplainer wrapper from scalablexplain.

Dependencies
------------
• PySpark (Spark 3.x)
• SynapseML JAR(s) in ./jars/            ← copy or download first
• pandas, scikit-learn
• scalablexplain package on PYTHONPATH   ← provides SparkLIMEExplainer
"""

import os
import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer

# ---------------------------------------------------------------------
# 1. Build Spark session that loads SynapseML jars
# ---------------------------------------------------------------------
from pyspark.sql import SparkSession

def build_spark_session(app_name: str = "Spark LIME Demo") -> SparkSession:

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[2]")               # adjust cores as needed
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.11") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ---------------------------------------------------------------------
# 2. Load data & train a PySpark model
# ---------------------------------------------------------------------
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

def train_model(spark: SparkSession):
    data = load_breast_cancer()
    pdf = pd.DataFrame(data.data, columns=data.feature_names)
    pdf["label"] = data.target

    sdf = spark.createDataFrame(pdf)
    input_cols = list(data.feature_names)

    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[assembler, lr])

    model = pipeline.fit(sdf)
    return model, sdf, input_cols


# ---------------------------------------------------------------------
# 3. Run LIME explainer
# ---------------------------------------------------------------------
def run_lime(model, sdf, input_cols):
    try:
        from scalablexplain.lime.spark_cluster import SparkLIMEExplainer
    except ImportError as e:
        sys.exit("[ERROR] Cannot import SparkLIMEExplainer – "
                 "make sure scalablexplain is on PYTHONPATH") 
    explainer = SparkLIMEExplainer(
        model=model,
        input_cols=input_cols,
        target_col="probability",
        target_classes=[1],   # explain the positive class
        num_samples=5000
    )
    explainer.build_explainer(sdf)

    # Explain the first 10 rows
    result = explainer.explain(sdf.limit(10))

    print("[INFO] LIME results (first few rows):")
    result.select("limeValues").show(truncate=False)


# ---------------------------------------------------------------------
# 4. Main entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    spark = build_spark_session()
    try:
        model, sdf, input_cols = train_model(spark)
        run_lime(model, sdf, input_cols)
    finally:
        spark.stop()
