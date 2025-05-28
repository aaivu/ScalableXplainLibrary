# test/test_lime_explainer.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from explainers.lime_explainer import LIMEExplainer

def test_lime_explainer_row_and_batch():
    # Create dummy data
    X = pd.DataFrame(np.random.rand(100, 4), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] + X["f1"] > 1).astype(int)
    model = RandomForestClassifier().fit(X, y)

    explainer = LIMEExplainer(model, X)
    
    instance = X.iloc[[0]]
    row_result = explainer.explain_row(instance)
    assert hasattr(row_result, 'as_list')  # LIME explanation object

    batch_result = explainer.explain(X.head(5))
    assert isinstance(batch_result, list)
    assert hasattr(batch_result[0], 'as_list')
