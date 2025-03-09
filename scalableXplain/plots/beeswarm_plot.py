from .abstract_plot import AbstractPlot
import shap

class BeeSwarmPlot(AbstractPlot):
    """
    Generates a beeswarm plot for SHAP values.
    """

    def render(self, shap_values, save_path="beeswarm_plot.png"):
        shap.plots.beeswarm(shap_values)
        shap.plt.savefig(save_path)
        shap.plt.show()
