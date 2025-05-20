import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from explainers.lime_explainer import LIMEExplainer

def test_lime_on_pandas():
    X = pd.DataFrame(np.random.rand(100, 4), columns=["x1", "x2", "x3", "x4"])
    y = (X["x1"] + X["x2"] > 1).astype(int)

    model = RandomForestClassifier()
    model.fit(X, y)

    explainer = LIMEExplainer(model, X)
    explanation = explainer.explain(X.iloc[[0]])

    assert hasattr(explanation, 'as_list')
