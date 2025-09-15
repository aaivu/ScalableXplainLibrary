'''
DistributedIMMExplainer provides a Python wrapper for distributed Iterative Mistake Minimization(IMM) tree extraction from kmeans clustered Spark DataFrames. 
It leverages a JVM-based Scala IMM implementation to generate interpretable decision trees from cluster centers and Spark ML KMeans results. 
The class supports exporting the tree structure, parsing Scala output, and visualizing the tree in Graphviz format.

Attributes:
    kmeans_model: Trained pyspark.ml.clustering.KMeansModel containing cluster centers.
    num_splits: Number of splits to consider for IMM tree construction (controls tree granularity).
    max_bins: Maximum number of bins for feature discretization in IMM.
    seed: Random seed for reproducibility.

Methods:
    __init__(kmeans_model, num_splits=32, max_bins=32, seed=42):
        Initializes the DistributedIMMExplainer with a Spark KMeans model and IMM tree parameters.

    explain(clustered_df):
        Runs IMM tree extraction on a Spark DataFrame containing cluster assignments and features.
        Returns the raw JVM tree nodes, split thresholds, and a Scala-formatted tree string.

    print_tree(tree_string):
        Prints the IMM tree structure as formatted by the Scala backend.

    parse_tree_string(tree_string):
        Parses the Scala tree string into a Python dictionary of nodes, extracting node attributes.

    export_graphviz(tree_string, output_path="imm_tree.dot"):
        Converts the parsed IMM tree into Graphviz .dot format for visualization, saving to file.
'''

from pyspark.ml.clustering import KMeansModel
from pyspark.sql import DataFrame
from typing import Tuple, List, Any
import re


class DistributedIMMExplainer:
    def __init__(self, kmeans_model: KMeansModel, num_splits: int = 32, max_bins: int = 32, seed: int = 42):
        self.kmeans_model = kmeans_model
        self.num_splits = num_splits
        self.max_bins = max_bins
        self.seed = seed

    def explain(self, clustered_df: DataFrame) -> Tuple[List[Any], List[List[Any]], str]:
        """
        clustered_df: must include 'features' (Vector) and 'prediction' (Int)
        Returns:
            - raw tree (list of JVM nodes)
            - splits (list of thresholds)
            - tree_string (Scala TreePrinter output)
        """
        spark = clustered_df.sparkSession
        sc = spark.sparkContext
        gateway = sc._gateway
        imm_wrapper = gateway.jvm.dimm.driver.IMMWrapper

        # Convert cluster centers to java.util.List[Vector]
        centers = self.kmeans_model.clusterCenters()
        java_centers = gateway.jvm.java.util.ArrayList()
        for c in centers:
            java_double_array = gateway.new_array(gateway.jvm.double, len(c))
            for i, val in enumerate(c):
                java_double_array[i] = float(val)
            java_vector = gateway.jvm.org.apache.spark.ml.linalg.Vectors.dense(java_double_array)
            java_centers.add(java_vector)

        # Call Scala
        result_map = imm_wrapper.runIMMFromClusteredDF(
            clustered_df._jdf,
            java_centers,
            self.num_splits,
            self.max_bins,
            self.seed
        )

        tree = list(result_map.get("tree"))
        splits = [list(s) for s in list(result_map.get("splits"))]
        tree_string = result_map.get("tree_string")

        return tree, splits, tree_string

    def print_tree(self, tree_string: str):
        """Print the IMM tree from Scala output."""
        print("\n=== IMM Tree ===")
        print(tree_string)

    def parse_tree_string(self, tree_string: str) -> dict:
        """Parse Scala-formatted tree string to a Python dict of nodes."""
        nodes = {}
        for line in tree_string.strip().split("\n"):
            if not line.startswith("Node"):
                continue
            match = re.match(
                r"Node (\d+): (Leaf|Node \d+) \| depth=(\d+) \| split=([^\|]+) \| clusters=\[(.*?)\] \| samples=(\d+) \| mistakes=(\d+)",
                line)
            if not match:
                continue
            node_id = int(match.group(1))
            label = match.group(2)
            depth = int(match.group(3))
            split = match.group(4).strip()
            clusters = match.group(5).split(",") if match.group(5) else []
            samples = int(match.group(6))
            mistakes = int(match.group(7))
            nodes[node_id] = {
                "id": node_id,
                "isLeaf": (label == "Leaf"),
                "depth": depth,
                "split": split,
                "clusters": clusters,
                "samples": samples,
                "mistakes": mistakes
            }
        return nodes

    def export_graphviz(self, tree_string: str, output_path="imm_tree.dot"):
        """Parse and export the tree to Graphviz .dot format."""
        nodes = self.parse_tree_string(tree_string)
        with open(output_path, "w") as f:
            f.write("digraph IMMTree {\n")
            for node_id, node in nodes.items():
                label = f"depth={node['depth']}\\nclusters={','.join(node['clusters'])}\\n{node['split']}\\nsamples={node['samples']}, mistakes={node['mistakes']}"
                f.write(f'  {node_id} [label="{label}"];\n')
            # Add edges (binary layout assumption: left = 2i+1, right = 2i+2)
            for node_id in nodes:
                left = 2 * node_id + 1
                right = 2 * node_id + 2
                if left in nodes:
                    f.write(f"  {node_id} -> {left};\n")
                if right in nodes:
                    f.write(f"  {node_id} -> {right};\n")
            f.write("}\n")
