import os
import pytest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

from scalablexplain.imm.dimm.pydimm import PyIMM  # Adjust this import if wrapper is elsewhere


@pytest.mark.spark
def test_pyimm_run():
    # === Setup Spark with IMM JAR ===
    jars_dir = os.path.join(os.getcwd(), "jars")
    jar_files = [os.path.join(jars_dir, f) for f in os.listdir(jars_dir) if f.endswith(".jar")]
    assert any("IMM" in j for j in jar_files), "IMM backend JAR not found"
    imm_jar = next(j for j in jar_files if "IMM" in j)

    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Test PyIMM") \
        .config("spark.jars", imm_jar) \
        .getOrCreate()

    try:
        print(spark._jvm)  # prints something like <py4j.java_gateway.JavaPackage object ...>
        print(dir(spark._jvm.dimm))       # should show `wrapper`
        print(dir(spark._jvm.dimm.wrapper))  # should show `IMMWrapper`
        print(spark._jvm.dimm.wrapper.IMMWrapper)  # ← if this prints <JavaPackage>, it's not loaded
    except:
        print('Issue Occurred')

    # === Generate mock clustered data ===
    np.random.seed(42)
    k = 3
    points_per_cluster = 10
    d = 2  # number of features

    centers = np.array([[1, 1], [5, 5], [9, 1]])
    data = []
    for i, c in enumerate(centers):
        for _ in range(points_per_cluster):
            pt = np.random.normal(loc=c, scale=0.5)
            data.append((float(i), pt.tolist()))  # (cluster_label, features)

    # Create Spark DataFrame
    pdf = pd.DataFrame(data, columns=["cluster", "features_raw"])
    pdf[["x1", "x2"]] = pd.DataFrame(pdf["features_raw"].to_list(), index=pdf.index)
    sdf = spark.createDataFrame(pdf[["cluster", "x1", "x2"]])

    assembler = VectorAssembler(inputCols=["x1", "x2"], outputCol="features")
    vec_df = assembler.transform(sdf).select("cluster", "features")

    # === Run IMM Wrapper ===
    imm = PyIMM(spark, jar=imm_jar)
    result = imm.run(vec_df, centers)

    # === Check result ===
    assert "tree" in result
    assert "splits" in result
    assert isinstance(result["tree"], dict)
    assert isinstance(result["splits"], list)

    spark.stop()
