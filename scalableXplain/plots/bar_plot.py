import numpy as np
import matplotlib.pyplot as plt
from .abstract_plot import AbstractPlot

class BarPlot(AbstractPlot):
    """
    Creates a SHAP-like bar plot for feature importances.
    """

    def render(self, save_path="feature_importance.png"):
        features, importances = zip(*self.data.items())

        # Convert multi-class importance to a single value
        if isinstance(importances[0], (list, np.ndarray)):
            importances = [float(np.mean(imp)) for imp in importances]

        # Sort features by importance (Descending)
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = np.array(features)[sorted_idx]
        sorted_importances = np.array(importances)[sorted_idx]

        plt.figure(figsize=(8, 6))
        plt.barh(sorted_features, sorted_importances, color='#ff0151')

        # Add labels next to bars
        for i, v in enumerate(sorted_importances):
            plt.text(v + 0.01, i, f"+{v:.2f}", va="center", fontsize=10, color="black")

        plt.xlabel("mean(|value|)", fontsize=12)
        plt.ylabel("")
        plt.title("Feature Importance", fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
