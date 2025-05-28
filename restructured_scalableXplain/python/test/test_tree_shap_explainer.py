import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from explainers.tree_shap import TreeSHAPExplainer

def test_tree_shap_on_pandas():
    X = pd.DataFrame(np.random.rand(100, 4), columns=["f1", "f2", "f3", "f4"])
    y = (X["f1"] + X["f2"] > 1).astype(int)

    model = RandomForestClassifier()
    model.fit(X, y)

    explainer = TreeSHAPExplainer(model, X)
    shap_vals = explainer.explain(X.iloc[[0]])

    # Validate output type and shape
    assert isinstance(shap_vals, (list, np.ndarray))
    if isinstance(shap_vals, list):
        assert len(shap_vals[0]) == X.shape[1]
    elif isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 2:
            # (samples, features)
            assert shap_vals.shape[1] == X.shape[1]
        elif shap_vals.ndim == 3:
            # (samples, features, classes)
            assert shap_vals.shape[1] == X.shape[1]
# test/test_tree_shap_explainer.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from explainers.tree_shap import TreeSHAPExplainer

def test_tree_shap_explainer_row_and_batch():
    # Create dummy data
    X = pd.DataFrame(np.random.rand(100, 4), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] + X["f1"] > 1).astype(int)
    model = RandomForestClassifier().fit(X, y)

    explainer = TreeSHAPExplainer(model, X)
    
    instance = X.iloc[[0]]
    row_result = explainer.explain_row(instance)
    assert isinstance(row_result, list) or isinstance(row_result, np.ndarray)

    batch_result = explainer.explain(X.head(10))
    assert isinstance(batch_result, list) or isinstance(batch_result, np.ndarray)
