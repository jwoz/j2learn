import numpy as np

class MaximumNode:
    def __init__(self, categories, underlying_nodes):
        self._categories = categories
        self._underlying_nodes = underlying_nodes

    def value(self):
        values = [node.value() for node in self._underlying_nodes]
        i = np.argmax(np.array(values))
        if values[i] == 0:
            return -1
        return self._categories[i]

    def derivative(self):
        return 0

    def probability(self):
        p = max([node.value() for node in self._underlying_nodes])
        return p

    def derivative_probability(self):
        """
        calculate wrt to largest value, but that's not strictly the whole truth.
        Need to track which weights derivative come into play.
        """
        values = [node.value() for node in self._underlying_nodes]
        i = np.argmax(np.array(values))
        jacobian = self._underlying_nodes(i).derivative()
