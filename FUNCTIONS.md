# ScalableXplain API Reference

This document describes the main APIs and methods available in the `scalableXplain` library, supporting both single-node Python and distributed Apache Spark environments for scalable explainable AI (XAI) on tabular data.

---

## Package Structure

```
scalableXplain/
├── src/
│   └── python/
│       └── scalablexplain/
│           ├── kernel_shap/
│           │   ├── single_node.py
│           │   └── spark_cluster.py
│           ├── lime/
│           │   ├── single_node.py
│           │   └── spark_cluster.py
│           └── iterative_mistake_minimization/
│               └── spark_cluster.py
├── notebooks/
├── tests/
```

---

## KernelSHAP

### Single Node API (`kernel_shap/single_node.py`)

**Class:** `KernelSHAPExplainer`

#### Methods

- `__init__(model, input_cols, background_data=None, num_samples=5000)`
  - Initializes the explainer with a trained sklearn-compatible model, a list of input feature columns, optional background data, and the number of samples for SHAP approximation.
  - `model`: Trained classifier or regressor.
  - `input_cols`: List of feature names.
  - `background_data`: DataFrame or array for SHAP background distribution.
  - `num_samples`: Controls SHAP sampling accuracy/speed.

- `build_explainer(training_data)`
  - Prepares the SHAP KernelExplainer using the provided training data.
  - If `background_data` is not set, samples 100 rows from `training_data` as background.
  - Automatically selects the correct prediction function (`predict_proba` for classification, `predict` for regression).

- `explain(instances_df)`
  - Computes SHAP values for the given instances.
  - Returns a DataFrame of SHAP values, with one row per instance and columns for each feature.

- `plot(shap_values_df, instances_df, max_instances=100)`
  - Visualizes SHAP values using bar plots (for single instance) or beeswarm plots (for multiple instances).
  - Saves the plot as a PNG file and displays it.

---

### Spark API (`kernel_shap/spark_cluster.py`)

**Class:** `SparkKernelSHAPExplainer`

#### Methods

- `__init__(model, input_cols, target_col="probability", target_classes=[1], background_data=None, num_samples=5000)`
  - Initializes the distributed SHAP explainer for Spark ML models.
  - `model`: Trained Spark ML model.
  - `input_cols`: List of feature names.
  - `target_col`: Output column to explain (e.g., "probability").
  - `target_classes`: List of class indices to explain.
  - `background_data`: Spark DataFrame for SHAP background.
  - `num_samples`: Number of samples for SHAP estimation.

- `build_explainer(training_data)`
  - Sets up SynapseML's TabularSHAP explainer using the provided training data and optional background data.
  - Broadcasts background data for efficiency.

- `explain(instances_df)`
  - Computes SHAP values for the given Spark DataFrame of instances.
  - Returns a Spark DataFrame with SHAP values.

- `plot(explained_df, original_df, max_instances=100)`
  - Collects SHAP values and original features from Spark DataFrames.
  - Visualizes SHAP values using bar or beeswarm plots.
  - Saves the plot as a PNG file and displays it.

---

## LIME

### Single Node API (`lime/single_node.py`)

**Class:** `LIMEExplainer`

#### Methods

- `__init__(model, input_cols, kernel_width=None, class_names=None, num_features=10, random_state=None)`
  - Initializes the LIME explainer for a local model and feature set.
  - `model`: Trained sklearn-compatible model.
  - `input_cols`: List of feature names.
  - `kernel_width`: Width of the kernel for weighting perturbed samples.
  - `class_names`: Optional list of class names for classification.
  - `num_features`: Number of features to include in the explanation.
  - `random_state`: Seed for reproducibility.

- `build_explainer(training_data)`
  - Prepares the LIME explainer using the provided training data.
  - Fits the explainer to the data distribution and sets up internal structures for perturbation and explanation.

- `explain(instances_df, labels=(1,), num_samples=5000)`
  - Generates LIME explanations for the given instances.
  - Perturbs each instance, fits a local surrogate model, and computes feature attributions.
  - Returns a DataFrame or dictionary with feature importances for each instance.

- `plot(explanations, instance_idx=0)`
  - Visualizes LIME explanations for one or more instances.
  - For a single instance, displays a bar chart of feature importances.
  - For multiple instances, can aggregate and display summary plots.

---

### Spark API (`lime/spark_cluster.py`)

**Class:** `SparkLIMEExplainer`

#### Methods

- `__init__(model, input_cols, kernel_width=None, class_names=None, num_features=10, random_state=None)`
  - Initializes the distributed LIME explainer for Spark ML models.
  - Accepts similar parameters as the single-node version.

- `build_explainer(training_data)`
  - Sets up SynapseML's TabularLIME explainer using the provided training data.
  - Broadcasts background data for distributed computation.

- `explain(instances_df, labels=(1,), num_samples=5000)`
  - Computes LIME explanations for the given Spark DataFrame.
  - Perturbs instances and fits local surrogate models in a distributed fashion.
  - Returns a Spark DataFrame with feature importances.

- `plot(explained_df, original_df, max_instances=100)`
  - Visualizes LIME explanations using bar or beeswarm plots.
  - Saves the plot as a PNG file and displays it.

---

## Iterative Mistake Minimization (IMM)

### Spark API (`imm/spark_cluster.py`)

**Class:** `DistributedIMMExplainer`

#### Description

`DistributedIMMExplainer` is a Python wrapper for distributed Iterative Mistake Minimization (IMM) tree extraction, designed for use with Spark DataFrames that have been clustered using Spark ML's KMeans. The class leverages a JVM-based Scala IMM implementation to generate interpretable decision trees from cluster centers and cluster assignments. It provides utilities for exporting, parsing, and visualizing the extracted tree structure.

#### Attributes

- **kmeans_model**: Trained `pyspark.ml.clustering.KMeansModel` containing cluster centers.
- **num_splits**: Number of splits to consider for IMM tree construction (controls tree granularity).
- **max_bins**: Maximum number of bins for feature discretization in IMM.
- **seed**: Random seed for reproducibility.

#### Methods

- `__init__(kmeans_model, num_splits=32, max_bins=32, seed=42)`
  - Initializes the `DistributedIMMExplainer` with a Spark KMeans model and IMM tree parameters.
  - Stores the clustering model and configuration for tree extraction.

- `explain(clustered_df)`
  - Runs IMM tree extraction on a Spark DataFrame containing cluster assignments and feature vectors.
  - Internally calls the Scala IMM implementation via PySpark's JVM gateway.
  - Converts cluster centers to JVM objects and passes them to the Scala backend.
  - Returns:
    - **tree**: List of raw JVM tree nodes.
    - **splits**: List of split thresholds used in the tree.
    - **tree_string**: Scala-formatted string representation of the tree (for printing and parsing).

- `print_tree(tree_string)`
  - Prints the IMM tree structure as formatted by the Scala backend.
  - Useful for quick inspection of the extracted tree in a human-readable format.

- `parse_tree_string(tree_string)`
  - Parses the Scala-formatted tree string into a Python dictionary of nodes.
  - Extracts node attributes such as node ID, leaf status, depth, split condition, clusters, sample count, and mistake count.
  - Enables programmatic analysis and manipulation of the tree structure in Python.

- `export_graphviz(tree_string, output_path="imm_tree.dot")`
  - Converts the parsed IMM tree into Graphviz `.dot` format for visualization.
  - Saves the tree structure to a file, allowing users to render and explore the tree graphically.
  - Assumes a binary tree layout for edge construction.

#### Usage Notes

- The IMM algorithm is implemented in Scala and accessed via the JVM gateway in PySpark.
- The input DataFrame must include a `features` column (Spark ML vector) and a `prediction` column (cluster assignment).
- The exported Graphviz file can be visualized using tools like `dot` or online Graphviz viewers.
- This API is intended for distributed workloads and requires the IMM Scala JAR to be included in the Spark session.

- **Example Notebooks:**  
  See the `notebooks/` folder for step-by-step demos of each API.

---

## References

- [SynapseML Documentation](https://microsoft.github.io/SynapseML/)
- [SHAP Documentation](https://shap.readthedocs.io/)