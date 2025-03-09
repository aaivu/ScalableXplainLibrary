from .abstract_plot import AbstractPlot
import shap

class TreePlot(AbstractPlot):
    """
    Generates a tree plot using SHAP.
    """

    def render(self, explainer, X, save_path="tree_plot.png"):
        shap.tree_plot(explainer, X)
        shap.plt.savefig(save_path)
        shap.plt.show()
