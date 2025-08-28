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