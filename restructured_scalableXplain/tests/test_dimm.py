import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

from scalablexplain.imm.dimm.pydimm import PyIMM  # Update if your class is elsewhere


def main():
    # === Locate IMM JAR ===
    # jars_dir = os.path.join(os.getcwd(), "jars")
    # jar_files = [os.path.join(jars_dir, f) for f in os.listdir(jars_dir) if f.endswith(".jar")]
    # imm_jar = next((j for j in jar_files if "IMM" in j), None)

    # if not imm_jar:
    #     print("❌ IMM backend JAR not found in ./jars/")
    #     return

    # print(f"✅ Using JAR: {imm_jar}")
    print(os.getcwd())
    # === Start Spark ===
    spark = SparkSession.builder \
        .appName("PyIMM Smoke Test") \
        .master("local[2]") \
        .config("spark.jars", "jars/Spark-DIMM-assembly-1.0.jar") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # === JVM Debug Info ===
    try:
        print("JVM: ", spark._jvm)
        print("Loaded packages: ", dir(spark._jvm.dimm.wrapper))
        # print("IMMWrapper: ", spark._jvm.dimm.wrapper.IMMWrapper$.MODULE$)
    except Exception as e:
        print("⚠️ JVM issue:", e)

    # === Generate clustered data ===
    np.random.seed(42)
    centers = np.array([[1, 1], [5, 5], [9, 1]])
    points_per_cluster = 10
    rows = []

    for i, center in enumerate(centers):
        for _ in range(points_per_cluster):
            pt = np.random.normal(loc=center, scale=0.5)
            rows.append((float(i), pt[0], pt[1]))

    pdf = pd.DataFrame(rows, columns=["cluster", "x1", "x2"])
    sdf = spark.createDataFrame(pdf)

    # === Assemble feature vectors ===
    assembler = VectorAssembler(inputCols=["x1", "x2"], outputCol="features")
    vec_df = assembler.transform(sdf).select("cluster", "features")

    # === Run IMM ===
    imm = PyIMM(spark)
    result = imm.run(vec_df, centers)

    print("\n=== IMM Output ===")
    print("Tree:")
    print(result["tree"])
    print("Splits:")
    print(result["splits"])

    spark.stop()


if __name__ == "__main__":
    main()
