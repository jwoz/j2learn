import math

import numpy as np


class SoftMaxNode:
    def __init__(self, categories, weights, underlying_nodes):
        self._categories = categories
        self._weights = weights
        self._underlying_nodes = underlying_nodes
        self._category_count = len(self._categories)
        self._underlying_node_count = len(self._underlying_nodes)

    def weights(self):
        return self._weights

    def weight_count(self):
        return len(self._weights)

    def set_weights(self, weights):
        assert len(weights) == len(self._weights)
        for w, ww in zip(self._weights, weights):
            w.set_weight(ww)

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
        return values

    def derivative(self, cache=None):
        """
        calculate wrt to largest value, but that's not strictly the whole truth.
        Need to track which weights derivative come into play.
        """
        values = self.value(cache)
        jacobian = []
        for kp in range(self._category_count):
            for i in range(self._underlying_node_count):
                derivatives = []
                for k in range(self._category_count):
                    s = values[k]
                    x = self._underlying_nodes[i].value(cache)
                    d = (x - 1.0) * s if kp == k else -s
                    derivatives.append(d)
                jacobian.append(derivatives)
        return jacobian

    def chain_rule_factors(self, cache=None):
        """
        :return: chain rule factors of node with respect to underlying weights (aka derivatives wrt underlying nodes)
        """
        values = self.value(cache)
        factors = []
        for k in range(self._category_count):
            node_factors = []
            for i in range(self._underlying_node_count):  # range(c*self._node_count, c*self._node_count+1):
                factor = values[k] * self._weights[k * self._underlying_node_count + i].weight()
                node_factors.append(factor)
            factors.append(node_factors)
        return factors

    def set_weight_derivatives(self, derivatives):
        for w, d in zip(self._weights, derivatives):
            w.set_derivative(d, is_list=True)
