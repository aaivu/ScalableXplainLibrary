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
