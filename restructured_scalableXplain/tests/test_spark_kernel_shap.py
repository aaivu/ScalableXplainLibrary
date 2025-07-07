# import pytest
# import os

# try:
#     from pyspark.sql import SparkSession
#     from pyspark.ml.classification import LogisticRegression
#     from pyspark.ml.feature import VectorAssembler
#     from pyspark.ml import Pipeline
#     # from synapse.ml.explainers import TabularSHAP
#     HAVE_SPARK = True
# except ImportError:
#     HAVE_SPARK = False

# pytestmark = pytest.mark.skipif(not HAVE_SPARK, reason="PySpark or SynapseML not available")

# import pandas as pd
# from sklearn.datasets import load_breast_cancer
# # from scalablexplain.kernel_shap.spark_cluster import SparkKernelSHAPExplainer


# def test_kernel_shap_spark_cluster():
#     # Define the full path to the local jars directory
#     jars_dir = os.path.join(os.getcwd(), "jars")

#     # Collect all jar files in that directory
#     jar_files = [os.path.join(jars_dir, f) for f in os.listdir(jars_dir) if f.endswith(".jar")]

#     # Join them into a comma-separated string
#     jars_str = ",".join(jar_files)

#     print(f"JARS STRING: {jars_str} ")
#     # Spark session
#     spark = SparkSession.builder \
#         .appName("Spark SHAP Test") \
#         .master("local[2]") \
#         .config("spark.jars", jars_str) \
#         .getOrCreate()
    
#     from scalablexplain.kernel_shap.spark_cluster import SparkKernelSHAPExplainer
#     _ = spark._jvm.com.microsoft.azure.synapse.ml.explainers.TabularSHAP
#     # Load dataset
#     data = load_breast_cancer()
#     pdf = pd.DataFrame(data.data, columns=data.feature_names)
#     pdf["label"] = data.target
#     sdf = spark.createDataFrame(pdf)
#     input_cols = data.feature_names.tolist()

#     # Train PySpark model
#     assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
#     lr = LogisticRegression(featuresCol="features", labelCol="label")
#     pipeline = Pipeline(stages=[assembler, lr])
#     model = pipeline.fit(sdf)

#     # Run explainer
#     explainer = SparkKernelSHAPExplainer(
#         model=model,
#         input_cols=input_cols,
#         target_col="probability",
#         target_classes=[1],
#         num_samples=100
#     )
#     explainer.build_explainer(sdf)
#     result = explainer.explain(sdf.limit(10))

#     # Check result
#     assert "shapValues" in result.columns
#     assert result.count() == 10

#     spark.stop()
