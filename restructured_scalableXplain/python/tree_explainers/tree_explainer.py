import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from abc import ABC, abstractmethod
from explainers.base import BaseExplainer

class TreeExplainer(BaseExplainer, ABC):
    def __init__(self, model, data, task_type="clustering"):
        super().__init__(model, data, task_type)

    @abstractmethod
    def explain_tree(self):
        """Returns the decision tree structure."""
        pass

    @abstractmethod
    def explain_instance(self, instance):
        """Returns the decision path or explanation for a single instance."""
        pass

    @abstractmethod
    def score(self, data):
        """Returns the true k-means cost."""
        pass

    @abstractmethod
    def surrogate_score(self, data):
        """Returns the surrogate cost based on cluster centers."""
        pass
