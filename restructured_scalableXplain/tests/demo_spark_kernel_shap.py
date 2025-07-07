#!/usr/bin/env python3
"""
spark_kernel_shap_demo.py
-------------------------------------------------
Train a simple logistic-regression model on the
Breast-Cancer-Wisconsin dataset with PySpark and
explain it using SynapseML’s Tabular SHAP via the
SparkKernelSHAPExplainer wrapper from
scalablexplain.

Requirements
------------
* PySpark (tested with Spark 3.x)
* SynapseML JAR(s) in ./jars/
* scikit-learn, pandas
* scalablexplain package in PYTHONPATH
"""

import os
import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer

# ---------------------------------------------------------------------
# 1. Build / verify the Spark session
# ---------------------------------------------------------------------
from pyspark.sql import SparkSession

def build_spark_session(app_name: str = "Spark SHAP Demo") -> SparkSession:
    # jars_dir = os.path.join(os.getcwd(), "jars")
    # if not os.path.isdir(jars_dir):
    #     sys.exit(f"[ERROR] Expected JARs in {jars_dir}. Copy SynapseML jars first.")
    # jar_files = [os.path.join(jars_dir, f) for f in os.listdir(jars_dir) if f.endswith(".jar")]
    # if not jar_files:
    #     sys.exit("[ERROR] No *.jar files found in ./jars/")

    # jars_str = ",".join(jar_files)
    # print(f"[INFO] Using JARs: {jars_str}")

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
# 2. Load data and train a PySpark model
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
# 3. Run Kernel SHAP explainer
# ---------------------------------------------------------------------
def run_shap(model, sdf, input_cols):
    # Make sure scalablexplain is importable
    try:
        from scalablexplain.kernel_shap.spark_cluster import SparkKernelSHAPExplainer
    except ImportError as e:
        sys.exit(f"[ERROR] Cannot import SparkKernelSHAPExplainer – "
                 "check that scalablexplain is on PYTHONPATH :{e}")

    explainer = SparkKernelSHAPExplainer(
        model=model,
        input_cols=input_cols,
        target_col="probability",
        target_classes=[1],   # explain the positive class
        num_samples=100       # <-- tune for fidelity vs. speed
    )
    explainer.build_explainer(sdf)
    result = explainer.explain(sdf.limit(10))

    print("[INFO] SHAP results (first few rows):")
    result.select("shapValues").show(truncate=False)


# ---------------------------------------------------------------------
# 4. Main entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    spark = build_spark_session()
    try:
        model, sdf, input_cols = train_model(spark)
        run_shap(model, sdf, input_cols)
    finally:
        spark.stop()
