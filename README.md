# ScalableXplain - Efficient XAI Tool for Scalable Machine Learning Models

![project] ![research]

- **Project Lead(s) / Mentor(s)**
  1. Dr. Uthayasanker Thayasivam  
  2. Dr. Sumanaruban Rajadurai  

- **Contributor(s)**
  1. Inuka Ampavila  
  2. Saadha Salim  
  3. Bojitha Liyanage  

**Useful Links**

- GitHub: [ScalableXplainLibrary](https://github.com/inuka-00/ScalableXplainLibrary)  
- Distributed IMM GitHub Repository: [distributed_imm](https://github.com/aaivu/distributed_imm)

---

## Summary

ScalableXplain is a unified, efficient, and scalable Explainable AI (XAI) library designed to work seamlessly with both single-node and distributed machine learning environments. The project aims to make interpretation of complex models practical at scale—bridging the gap between model accuracy and human understanding.

This tool integrates multiple explanation techniques, including SHAP (KernelSHAP and TreeSHAP), LIME, and IMM (Iterative Mistake Minimization), under a single interface that automatically detects the runtime environment (e.g., pandas or PySpark). With distributed support via Apache Spark, ScalableXplain empowers practitioners to interpret large-scale models trained on massive datasets, without compromising on performance or usability.

---

## Description

ScalableXplain is developed to address a key challenge in modern machine learning workflows: how to generate meaningful explanations at scale. While traditional XAI tools work well on small datasets, they struggle with the size and complexity of modern pipelines. ScalableXplain introduces a modular and extensible library that supports:

- Local (single-node) explainers using NumPy, pandas, and scikit-learn  
- Distributed explainers using PySpark and SynapseML  
- Seamless switching between environments through unified APIs

## Distributed Iterative Mistake Minimization

**D-IMM** is a scalable and interpretable algorithm for explaining **k-means clustering** results using decision trees. Built for distributed environments, D-IMM extends the original **Iterative Mistake Minimization (IMM)** algorithm to handle large-scale datasets with millions of instances efficiently using Apache Spark.
This is a novel algorithm presented by us in this package through our research. It is based on the IMM algorithm introduced in https://arxiv.org/abs/2002.12538.
<img width="645" height="762" alt="image" src="https://github.com/user-attachments/assets/3dc24766-d0f1-4b32-bca5-1276c7c49efd" />

---

## 🔍 Overview

Traditional clustering methods like k-means are powerful but hard to interpret—especially on large datasets. D-IMM bridges this gap by constructing **human-readable decision trees** that approximate the original clustering assignments. It provides **global, post-hoc explanations** that scale seamlessly with data volume and dimensionality.

---

## ✨ Key Features

- ✅ **Scalable to 10M+ records** using distributed Spark execution.
- ✅ **Faithful explanations** that minimize mismatches with k-means.
- ✅ **Histogram-based binning** for fast, repeatable split evaluations.
- ✅ **Distributed mistake counting** and node refinement loop.
- ✅ Produces interpretable **global decision trees**.

---

## 📈 Performance Highlights

- Achieves up to **3.2× speedup** compared to single-node IMM.
- Preserves or improves **clustering fidelity** (mistake %, surrogate cost).
- Demonstrates **linear scalability** with increasing Spark executors.
- Tested on real-world datasets like **HIGGS (11M points)** and **SUSY**.

---

## 🛠 Built With

- Apache Spark 3.5.x
- Scala 2.12
- Java 17
- Compatible with PySpark via wrapper interface

---

### Project Phases

1. **Exploration & Benchmarking**
   - Surveyed existing explanation techniques and frameworks
   - Ran initial experiments on SHAP, LIME, and IMM for synthetic and benchmark datasets
  
2. **Research on creating a scalable algorithm that is equivalent to IMM**
   - Researched existing algorithms for threshold tree building in Apache Spark
   - Introduced histogram based candidate split discovery and histogram based mistake calculation to the existing IMM algorithm to make it scalable and efficient
  
3. **Experiments and Testing**
   - Tested the novel algorithm on a set of large scale datasets to verify scalability and validity of results

4. **Implementation of Package**
   - Developed unified wrapper classes for SHAP, LIME, and IMM  
   - Implemented both single-node and distributed versions  
   - Created automatic backend detection and dispatch  

5. **Optimization & Visualization**
   - Added visual support: SHAP bar plots, beeswarm plots, LIME text highlights  
   - Implemented efficient histogram-based mistake calculations for IMM  
   - Optimized runtime and memory for large datasets using Spark  

6. **Integration & Packaging**
   - Integrated Scala-based D-IMM using Py4J bridge  
   - Packaged the system as a pip-installable module  
   - Added command-line utilities and Jupyter notebook demos  


---

## More References

1. [SynapseML - Microsoft Distributed Machine Learning Toolkit](https://github.com/microsoft/SynapseML)  
2. [Original IMM Paper (Iterative Mistake Minimization)](https://arxiv.org/abs/2006.01275)

---

### License

Apache License 2.0

### Code of Conduct

Please read our [code of conduct document here](https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md).

[project]: https://img.shields.io/badge/-Project-blue  
[research]: https://img.shields.io/badge/-Research-yellowgreen
