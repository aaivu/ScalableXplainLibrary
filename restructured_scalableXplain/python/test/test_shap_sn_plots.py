# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from explainers.tree_shap import TreeSHAPExplainer

# def test_tree_shap_waterfall_plot_saves(tmp_path):
#     X = pd.DataFrame(np.random.rand(100, 3), columns=["a", "b", "c"])
#     y = (X["a"] + X["b"] > 1).astype(int)
#     model = RandomForestClassifier().fit(X, y)

#     explainer = TreeSHAPExplainer(model, X)
#     instance = X.iloc[[0]]
#     print(type(instance))
#     print(instance)
#     explanation = explainer.explain(instance)

#     # Plot and save
#     explainer.plot(explanation, instance=instance)
#     out_file = tmp_path / "waterfall_plot.png"
#     plt.savefig(out_file)
#     plt.close()

#     assert out_file.exists()
#     assert out_file.stat().st_size > 0

# def test_tree_shap_beeswarm_plot_saves(tmp_path):
#     X = pd.DataFrame(np.random.rand(100, 3), columns=["a", "b", "c"])
#     y = (X["a"] + X["b"] > 1).astype(int)
#     model = RandomForestClassifier().fit(X, y)

#     explainer = TreeSHAPExplainer(model, X)
#     batch = X.iloc[:10]
#     print(type(batch))
#     print(batch)
#     explanation = explainer.explain(batch)

#     # Plot and save
#     explainer.plot(explanation, instance=batch)
#     out_file = tmp_path / "beeswarm_plot.png"
#     plt.savefig(out_file)
#     plt.close()

#     assert out_file.exists()
#     assert out_file.stat().st_size > 0
