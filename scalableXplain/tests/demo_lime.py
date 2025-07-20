#!/usr/bin/env python3
"""
lime_demo.py
-------------------------------------------------
Train a logistic regression model on the Breast-Cancer-Wisconsin dataset
and explain it using LIME via the LIMEExplainer class from scalablexplain.

Requirements
------------
• scikit-learn
• pandas, matplotlib, seaborn
• lime
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, X_train_scaled, X_test_scaled, data.feature_names.tolist(), data.target_names.tolist()


def run_lime(model, X_train, X_test, input_cols, class_names):
    try:
        from scalablexplain.lime.single_node import LIMEExplainer
    except ImportError as e:
        sys.exit(f"[ERROR] Cannot import LIMEExplainer – "
                 f"check that scalablexplain is on PYTHONPATH: {e}")

    explainer = LIMEExplainer(
        model=model,
        input_cols=input_cols,
        class_names=class_names,
        mode="classification",
        target_classes=[1],  # Explain the positive class
        background_data=X_train.sample(n=100, random_state=42),
        num_samples=1000
    )

    explainer.build_explainer(X_train)

    # -------- Beeswarm Plot for 50 Points --------
    print("[INFO] Explaining 50 samples...")
    subset_df = X_test.head(50)
    lime_df = explainer.explain(subset_df)
    print("[INFO] Plotting beeswarm LIME values...")
    explainer.plot(lime_df, original_df=subset_df, max_instances=50)

    # -------- Bar Plot for Single Instance --------
    print("[INFO] Explaining a single instance...")
    single_df = X_test.head(1)
    lime_single_df = explainer.explain(single_df)
    print("[INFO] Plotting bar chart for single instance...")
    explainer.plot(lime_single_df, original_df=single_df, max_instances=1)


if __name__ == "__main__":
    model, X_train, X_test, input_cols, class_names = train_model()
    run_lime(model, X_train, X_test, input_cols, class_names)
