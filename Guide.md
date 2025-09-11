# ScalableXplain Library Structure and Usage Guide

## Overview

ScalableXplain is an extensible library for scalable explainable AI (XAI) on tabular data. It provides unified APIs for popular model explanation algorithms, supporting both single-node (Python) and distributed (Apache Spark) environments.

---

## Supported Algorithms

- **KernelSHAP**  
  SHAP (SHapley Additive exPlanations) using KernelExplainer for model-agnostic explanations.  
  - Available for both single-node Python and distributed Spark (via SynapseML).

- **LIME**  
  Local Interpretable Model-agnostic Explanations for tabular data.  
  - Available for both single-node Python and distributed Spark (via SynapseML).

- **Iterative Mistake Minimization (IMM) [Distributed]**  
  Custom distributed algorithm for explanation, designed for K-means clusters on apche spark.   
  - Requires custom Scala implementation and Spark integration.

---

## Architecture

- **Single Node (Python):**  
  - Algorithms are implemented in Python for use with scikit-learn and pandas.
  - Each algorithm has a dedicated file (e.g., `kernel_shap/single_node.py`, `lime/single_node.py`).

- **Distributed (Apache Spark):**  
  - Algorithms leveraging SynapseML are implemented in Python and require the SynapseML JAR.
  - Custom distributed algorithms (e.g., IMM) must be implemented in Scala, packaged as a JAR, and included in the Spark session.
  - Each algorithm has a dedicated file for Spark (e.g., `kernel_shap/spark_cluster.py`, `lime/spark_cluster.py`).

---

## Installation

- **Python Components:**  
  Install the Python package using pip:
  ```bash
  pip install scalablexplain
  ```

- **Spark Cluster Setup:**  
  - **SynapseML Algorithms:**  
    Include the SynapseML JAR when starting your Spark session.  
    Example (see notebooks for details):
    ```python
    .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.11")
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
    ```
  - **Custom Distributed Algorithms:**  
    Implement the algorithm in Scala, build the JAR, and include it in your Spark session:
    ```python
    .config("spark.jars", "/path/to/custom_algorithm.jar")
    ```

---

## File Structure

- `src/python/scalablexplain/algorithm_name/single_node.py`  
  Python implementation for single-node systems.

- `src/python/scalablexplain/algorithm_name/spark_cluster.py`  
  Python wrapper for distributed Spark systems (using SynapseML or custom JARs).

- `notebooks/`  
  Example Jupyter notebooks for both single-node and Spark usage.

- `tests/`  
  Demo scripts and unit tests.

---

## Usage

- Choose the appropriate API (single-node or Spark) based on your environment.
- For distributed workloads, ensure all required JARs are included in your Spark session.
- Refer to the example notebooks for step-by-step guides on using each algorithm.

---

## Notes

- SynapseML-based algorithms require the SynapseML JAR.
- Custom distributed algorithms require Scala implementation and JAR packaging.
- Python and Spark code are separated for

# Installation Guide

This section describes how to install and set up the ScalableXplain library for both Python and Scala components.

---

## Required Python Packages

You will need the following Python packages:

- `pyspark` (version should match your installed Spark, e.g., `pyspark>=3.5.1`)
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `lime`
- `shap`
- `synapseml`
- `seaborn`
- `graphviz`
- `setuptools`, `wheel` (for building the Python wheel)

You can install all dependencies with:

```bash
pip install pyspark numpy pandas scikit-learn matplotlib lime shap synapseml seaborn graphviz setuptools wheel
```

---

## Building the Python Wheel

The Python package is managed using a `pyproject.toml` file located at:

```
scalableXplain/scalableXplain/pyproject.toml
```

To build the wheel file, run the following command from the `scalableXplain/scalableXplain` directory:

```bash
python3 -m build
```
or, if you only want the wheel:
```bash
python3 -m build --wheel
```

This will generate a `.whl` file in the `dist/` directory, which you can install with:

```bash
pip install dist/scalablexplain-0.1.0-py3-none-any.whl
```

---

## Building the Scala JAR for DIMM

The Scala implementation for distributed IMM (Iterative Mistake Minimization) is managed using `sbt`.  
The `build.sbt` file should be located in the root of your Scala IMM project, for example:

```
scalableXplain/dimm-scala/build.sbt
```

To build the JAR file, navigate to the `dimm-scala` directory and run:

```bash
sbt package
```

The resulting JAR file will be found in:

```
scalableXplain/dimm-scala/target/scala-<scala_version>/
```
(e.g., `target/scala-2.12/dimm_2.12-0.1.jar`)

Include this JAR in your Spark session using the `spark.jars` configuration.

---

## File Locations

- **Python build configuration:**  
  `scalableXplain/scalableXplain/pyproject.toml`

- **Scala sbt build file:**  
  `scalableXplain/dimm-scala/build.sbt`

---

## Additional Notes

- Ensure the versions of `pyspark` and your Spark cluster match.
- For distributed algorithms using SynapseML, include the SynapseML JAR in your Spark session as shown in the example notebooks.
- For custom distributed algorithms (like IMM), include your built Scala JAR in the Spark