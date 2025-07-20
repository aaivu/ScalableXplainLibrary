from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from scalablexplain.kernel_shap.single_node import KernelSHAPExplainer


def test_kernel_shap_single_node():
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Build and run explainer
    explainer = KernelSHAPExplainer(clf, input_cols=X.columns.tolist(), num_samples=300)
    explainer.build_explainer(X_train)

    shap_df = explainer.explain(X_test.head(5))

    # Check shape and type
    assert isinstance(shap_df, pd.DataFrame)
    assert shap_df.shape == (5, len(X.columns))
    assert all(col in shap_df.columns for col in X.columns)
