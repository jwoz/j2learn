import math

import numpy as np


class SoftMaxNode:
    def __init__(self, categories, weights, underlying_nodes):
        self._categories = categories
        self._weights = weights
        self._underlying_nodes = underlying_nodes
        self._category_count = len(self._categories)
        self._underlying_node_count = len(self._underlying_nodes)

    @staticmethod
    def weights():
        return []

    def node_index(self, cache=None):
        if self in cache:
            return cache[self]
        values = self.value(cache)
        i = int(np.argmax(np.array(values)))
        return i

    def predict(self, cache=None):
        i = self.node_index(cache)
        return [self._categories[i]]

    def value(self, cache=None):
        values = []
        sum_weight = 0
        for c in range(self._category_count):
            exp_value = 0
            for n in range(self._underlying_node_count):  # range(c*self._node_count, c*self._node_count+1):
                weight = self._weights[c * self._underlying_node_count + n].weight()
                exp_value += self._underlying_nodes[n].value(cache) * self._weights[c * self._underlying_node_count + n].weight()
                sum_weight += weight
            exp_value = math.exp(exp_value)
            values.append(exp_value)
        values = [v / math.exp(sum_weight) for v in values]
        return valuesc /

    def derivative(self, cache=None):
        """
        calculate wrt to largest value, but that's not strictly the whole truth.
        Need to track which weights derivative come into play.
        """
        values = self.value(cache)

        jacobian = self._underlying_nodes[i].derivative(cache)
        return jacobian

    def chain_rule_factors(self, cache=None):
        """
        :return: chain rule factors of node with respect to underlying weights (aka derivatives wrt underlying nodes)
        """
        return

    def set_weight_derivatives(self, derivatives):
        pass
