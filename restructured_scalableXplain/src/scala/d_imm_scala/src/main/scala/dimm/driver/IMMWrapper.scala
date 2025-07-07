package dimm.driver

import dimm.core.{IMMRunner, Instance}
import dimm.tree.{Node, ContinuousSplit}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.JavaConverters._

object IMMWrapper {

  /** Run IMM when you already have a DF with
    *   • a **features** column  (org.apache.spark.ml.linalg.Vector)
    *   • a **prediction** column (Int: cluster id)
    * plus the K-means cluster centres.
    *
    * Everything else is delegated to `IMMRunner`.
    *
    * @return Java-serialisable map → safe across Py4J.
    */
  def runIMMFromClusteredDF(
      clustered: DataFrame,                       // ← already computed in PySpark
      clusterCenters: java.util.List[Vector],     // ← same order as prediction ids
      numSplits: Int  = 32,
      maxBins:  Int   = 32,
      seed:     Long  = 42L
  ): java.util.Map[String, Any] = {

    // ---------- 1.  Spark context ----------
    val spark = clustered.sparkSession
    import spark.implicits._

    // ---------- 2.  Convert DF → RDD[Instance] ----------
    val instRDD = clustered
      .select("features", "prediction")
      .rdd
      .map { case org.apache.spark.sql.Row(v: Vector, cid: Int) =>
        Instance(cid, 1.0, v)
      }
      .cache()

    // ---------- 3.  IMM ----------
    val (tree, splits) =
      IMMRunner.runIMM(instRDD,
                       clusterCenters.asScala.toArray,
                       numSplits,
                       maxBins,
                       seed)

    // ---------- 4.  Pack & return ----------
    java.util.Map.of(
      "tree",    tree.asJava,
      "splits",  splits.map(_.toSeq.asJava).toSeq.asJava
    )
  }
}
