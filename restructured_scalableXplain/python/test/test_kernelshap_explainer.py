# test/test_kernelshap_explainer.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from explainers.kernel_shap import KernelSHAPExplainer

def test_kernel_shap_explainer_row_and_batch():
    # Create synthetic dataset
    X = pd.DataFrame(np.random.rand(100, 3), columns=["a", "b", "c"])
    y = (X["a"] + X["b"] > 1).astype(int)
    model = LogisticRegression().fit(X, y)

    explainer = KernelSHAPExplainer(model, X)

    # Test row-level explanation
    instance = X.iloc[[0]]
    row_result = explainer.explain_row(instance)
    assert isinstance(row_result, list) or isinstance(row_result, np.ndarray)

    # Test batch-level explanation
    batch_result = explainer.explain(X.head(10))
    assert isinstance(batch_result, list) or isinstance(batch_result, np.ndarray)
