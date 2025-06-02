import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .ExKMC.ExKMC.Tree import Tree
from .ExKMC.ExKMC.Tree import convert_input
from .tree_explainer import TreeExplainer


class IMMExplainer(TreeExplainer):
    def __init__(self, model=None, data=None, k=3, max_leaves=None, verbose=0, base_tree="IMM", random_state=42):
        super().__init__(model, data, task_type="clustering")

        self.k = k
        self.verbose = verbose
        self.random_state = random_state

        # Build the tree
        self.tree = Tree(
            k=k,
            max_leaves=max_leaves,
            verbose=verbose,
            base_tree=base_tree,
            random_state=random_state
        )
        self.tree.fit(data, kmeans=model)

    def explain_tree(self):
        return self.tree.tree  # Returns root node (or a serializable structure)

    def explain_instance(self, instance):
        x = convert_input(instance)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._trace_path(self.tree.tree, x[0])

    def _trace_path(self, node, instance):
        """
        Recursively find the path to the leaf node and return the decision path.
        """
        path = []
        while not node.is_leaf():
            path.append((node.feature, node.value, instance[node.feature] <= node.value))
            if instance[node.feature] <= node.value:
                node = node.left
            else:
                node = node.right
        path.append(("Leaf", node.value))
        return path

    def score(self, data):
        return self.tree.score(data)

    def surrogate_score(self, data):
        return self.tree.surrogate_score(data)

    def plot(self, filename="imm_tree", feature_names=None, view=True):
        self.tree.plot(filename=filename, feature_names=feature_names, view=view)
