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

import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer

# ---------------------------------------------------------------------
# 1. Build / verify the Spark session
# ---------------------------------------------------------------------
from pyspark.sql import SparkSession

def build_spark_session(app_name: str = "Spark SHAP Demo") -> SparkSession:
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[2]")  # adjust cores as needed
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.11")
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
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
    try:
        from scalablexplain.kernel_shap.spark_cluster import SparkKernelSHAPExplainer
    except ImportError as e:
        sys.exit(f"[ERROR] Cannot import SparkKernelSHAPExplainer – "
                 f"check that scalablexplain is on PYTHONPATH: {e}")

    explainer = SparkKernelSHAPExplainer(
        model=model,
        input_cols=input_cols,
        target_col="probability",
        target_classes=[1],
        num_samples=100  # <-- tune for fidelity vs. performance
    )

    explainer.build_explainer(sdf)

    # -------- Beeswarm Plot for 50 Points --------
    print("[INFO] Explaining 50 samples...")
    subset_df = sdf.limit(50)
    result = explainer.explain(subset_df)

    print("[INFO] SHAP values (first few rows):")
    result.show(truncate=False)

    print("[INFO] Plotting beeswarm SHAP values...")
    explainer.plot(result, original_df=subset_df, max_instances=50)

    # -------- Bar Plot for Single Instance --------
    print("[INFO] Explaining a single instance...")
    single_df = sdf.limit(1)
    single_result = explainer.explain(single_df)

    print("[INFO] Plotting bar chart for single instance...")
    explainer.plot(single_result, original_df=single_df, max_instances=1)

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
