from itertools import cycle
import numpy as np


class SoftMaxNode:
    def __init__(self, categories, weights, underlying_nodes):
        self._categories = categories
        self._weights = weights
        self._underlying_nodes = underlying_nodes
        self._category_count = len(self._categories)
        self._node_count = len(self._underlying_nodes)

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
        values = []
        for c in range(self._category_count):
            exp_value = 0
            exp_weights = 0
            for n in range(self._node_count): # range(c*self._node_count, c*self._node_count+1):
                exp_value += self._underlying_nodes[n] * self._weights[c*self._node_count + n]
                exp_weights += self._underlying_nodes[n] * self._weights[c*self._node_count + n]

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
