import numpy as np


class MaximumNode:
    def __init__(self, categories, underlying_nodes):
        self._categories = categories
        self._underlying_nodes = underlying_nodes

    @staticmethod
    def count():
        return 0

    @staticmethod
    def weights():
        return []

    def node_index(self, cache=None):
        if self in cache:
            return cache[self]
        values = [node.value(cache) for node in self._underlying_nodes]
        i = int(np.argmax(np.array(values)))
        return i

    def predict(self, cache=None):
        i = self.node_index(cache)
        return [self._categories[i]]

    def value(self, cache=None):
        p = max([node.value(cache) for node in self._underlying_nodes])
        return [p]

    def derivative(self, cache=None):
        """
        calculate wrt to largest value, but that's not strictly the whole truth.
        Need to track which weights derivative come into play.
        """
        values = [node.value(cache) for node in self._underlying_nodes]
        i = np.argmax(np.array(values))
        jacobian = self._underlying_nodes[i].derivative(cache)
        return jacobian

    def set_weight_derivatives(self, derivatives):
        pass
