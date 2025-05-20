import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from explainers.kernel_shap import KernelSHAPExplainer

def test_kernel_shap_on_pandas():
    # Create dummy data
    X = pd.DataFrame(np.random.rand(100, 3), columns=["a", "b", "c"])
    y = (X["a"] + X["b"] > 1).astype(int)

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    # Initialize explainer and explain a single instance
    explainer = KernelSHAPExplainer(model, X)
    shap_vals = explainer.explain(X.iloc[[0]])

    # Check output type and shape
    assert isinstance(shap_vals, (list, np.ndarray))
    if isinstance(shap_vals, list):
        assert len(shap_vals[0]) == X.shape[1]
    elif isinstance(shap_vals, np.ndarray):
        assert shap_vals.shape[-1] == X.shape[1]
