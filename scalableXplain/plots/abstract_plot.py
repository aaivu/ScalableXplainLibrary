from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class AbstractPlot(ABC):
    """
    Abstract base class for all plots.
    """
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def render(self):
        """Render the plot."""
        pass
