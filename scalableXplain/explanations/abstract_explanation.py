from abc import ABC, abstractmethod

class AbstractExplanation(ABC):
    """
    Abstract base class for all explanation outputs.
    """

    @abstractmethod
    def to_dict(self):
        """Convert explanation to a dictionary format."""
        pass

    @abstractmethod
    def plot(self):
        """Plot the explanation using an appropriate visualization."""
        pass
