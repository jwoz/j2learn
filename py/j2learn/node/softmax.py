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

    def _exponentials(self, cache=None):
        exponentials = []
        for k in range(self._category_count):
            value = 0
            for i in range(self._underlying_node_count):
                weight = self._weights[k * self._underlying_node_count + i].weight()
                value += self._underlying_nodes[i].value(cache) * weight
            exponentials.append(math.exp(value))
        return exponentials

    def value(self, cache=None):
        if self in cache:
            return cache[self]
        exponentials = self._exponentials(cache)
        values = [e / sum(exponentials) for e in exponentials]
        cache[self] = values
        return values

    def derivative(self, cache=None):
        values = self.value(cache)
        jacobian = []
        exponentials = self._exponentials(cache)
        sum_exponentials = sum(exponentials)
        for kp in range(self._category_count):
            for i in range(self._underlying_node_count):
                derivatives = []
                for k in range(self._category_count):
                    x = self._underlying_nodes[i].value(cache)
                    d = (x * exponentials[k] * (sum_exponentials - exponentials[k])) / (sum_exponentials ** 2) if k == kp else -values[k] * values[kp] * x
                    derivatives.append(d)
                jacobian.append(derivatives)
        return jacobian

    def chain_rule_factors(self, cache=None):
        """
        :return: chain rule factors of node with respect to underlying weights (aka derivatives wrt underlying nodes)
        """
        factors = []
        exponentials = self._exponentials(cache)
        sum_exponentials = sum(exponentials)
        for k in range(self._category_count):
            derivatives = []
            for i in range(self._underlying_node_count):
                w = self._weights[k * self._underlying_node_count + i].weight()
                denomdx = sum([exponentials[j] * self._weights[j * self._underlying_node_count + i].weight() for j in range(self._category_count)])
                d = exponentials[k] * (w * sum_exponentials - denomdx) / (sum_exponentials ** 2)
                derivatives.append(d)
            factors.append(derivatives)
        return factors

    def set_weight_derivatives(self, derivatives):
        for w, d in zip(self._weights, derivatives):
            w.set_derivative(d, is_list=True)
