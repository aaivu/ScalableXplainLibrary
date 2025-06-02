import math

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from tree_explainers.imm_explainer import IMMExplainer

def test_imm_explainer_basic_functionality():
    # Generate synthetic clustering data
    X, _ = make_blobs(n_samples=300, centers=3, n_features=4, random_state=42)
    X = pd.DataFrame(X)
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)

    # Initialize and fit the explainer
    explainer = IMMExplainer(model=kmeans, data=X, k=3, verbose=0)

    # Test cost functions
    score = explainer.score(X)
    surrogate_score = explainer.surrogate_score(X)

    assert isinstance(score, float) and score >= 0
    assert isinstance(surrogate_score, float)
    assert math.isclose(surrogate_score, score, rel_tol=1e-9) or surrogate_score > score

    # Test explanation of a single instance
    instance = X.iloc[[0]]
    explanation_path = explainer.explain_instance(instance)

    assert isinstance(explanation_path, list)
    assert explanation_path[-1][0] == "Leaf"
    assert isinstance(explanation_path[-1][1], (int, np.integer))


    # Test plotting (side effect)
    try:
        filepath = "test/imm_plots/test_plot"
        explainer.plot(filename=filepath, feature_names=["f1", "f2", "f3", "f4"], view=False)
    except Exception as e:
        assert False, f"Plotting failed with error: {e}"
