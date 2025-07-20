#!/usr/bin/env python3
"""
kernel_shap_demo.py
-------------------------------------------------
Train a logistic regression model on the Breast-Cancer-Wisconsin dataset
and explain it using SHAP KernelExplainer via the KernelSHAPExplainer class.

Requirements
------------
• scikit-learn
• shap
• pandas, matplotlib
• scalablexplain package on PYTHONPATH
"""

import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="label")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, X_train_scaled, X_test_scaled, data.feature_names.tolist()


def run_shap(model, X_train, X_test, input_cols):
    try:
        from scalablexplain.kernel_shap.single_node import KernelSHAPExplainer
    except ImportError as e:
        sys.exit(f"[ERROR] Cannot import KernelSHAPExplainer – "
                 f"check that scalablexplain is on PYTHONPATH: {e}")

    explainer = KernelSHAPExplainer(
        model=model,
        input_cols=input_cols,
        background_data=X_train.sample(n=100, random_state=42),
        num_samples=100  # tradeoff between speed and accuracy
    )

    explainer.build_explainer(X_train)

    # -------- Beeswarm Plot for 50 Points --------
    print("[INFO] Explaining 50 samples...")
    subset_df = X_test.head(50)
    shap_df = explainer.explain(subset_df)

    print("[INFO] Plotting beeswarm SHAP values...")
    explainer.plot(shap_values_df=shap_df, instances_df=subset_df, max_instances=50)

    # -------- Bar Plot for Single Instance --------
    print("[INFO] Explaining a single instance...")
    single_df = X_test.head(1)
    shap_single_df = explainer.explain(single_df)

    print("[INFO] Plotting bar chart for single instance...")
    explainer.plot(shap_values_df=shap_single_df, instances_df=single_df, max_instances=1)


if __name__ == "__main__":
    model, X_train, X_test, input_cols = train_model()
    run_shap(model, X_train, X_test, input_cols)
