package dimm.driver

import dimm.core.{IMMRunner, Instance}
import dimm.tree.{Node, ContinuousSplit}
import dimm.tree.TreePrinter

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.JavaConverters._

object IMMWrapper {

  def runIMMFromClusteredDF(
      clustered: DataFrame,
      clusterCenters: java.util.List[Vector],
      numSplits: Int = 32,
      maxBins: Int = 32,
      seed: Long = 42L
  ): java.util.Map[String, Any] = {

    val spark = clustered.sparkSession
    import spark.implicits._

    val instRDD = clustered
      .select("features", "prediction")
      .rdd
      .map { case org.apache.spark.sql.Row(v: Vector, cid: Int) =>
        Instance(cid, 1.0, v)
      }
      .cache()

    val (tree, splits) = IMMRunner.runIMM(
      instRDD,
      clusterCenters.asScala.toArray,
      numSplits,
      maxBins,
      seed
    )

    // Convert root node to JSON-like structure
    val root = tree(0)
    val treeAsJson = serializeTree(root)

    // Inside runIMMFromClusteredDF
    val treeString = TreePrinter.printTreeAsString(tree, splits)

    java.util.Map.of(
      "tree", tree.asJava,
      "splits", splits.map(_.toSeq.asJava).toSeq.asJava,
      "tree_string", treeString
    )
  }

  /** Recursively convert Node to nested java.util.Map for Py4J-friendly parsing */
  def serializeTree(node: Node): java.util.Map[String, Any] = {
    val map = new java.util.HashMap[String, Any]()
    map.put("id", node.id)
    map.put("depth", node.depth)
    map.put("isLeaf", node.isLeaf)
    map.put("clusterIds", node.clusterIds.toArray.map(_.asInstanceOf[Integer]).toList.asJava)
    map.put("samples", node.samples.map(Int.box).orNull)
    map.put("mistakes", node.mistakes.map(Int.box).orNull)

    node.split.foreach { s =>
      val sMap = new java.util.HashMap[String, Any]()
      sMap.put("featureIndex", s.featureIndex)
      sMap.put("threshold", s.threshold)
      map.put("split", sMap)
    }

    node.leftChild.foreach(lc => map.put("leftChild", serializeTree(lc)))
    node.rightChild.foreach(rc => map.put("rightChild", serializeTree(rc)))

    map
  }
}
