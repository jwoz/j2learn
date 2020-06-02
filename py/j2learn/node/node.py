import numpy as np

from j2learn.node.data import ZeroNode


class Node:
    multi = False
    def __init__(self, activation, weights, underlying_nodes, name=''):
        self._activation = activation
        self._weights = weights
        self._underlying_nodes = underlying_nodes
        self._name = name
        self._chain_rule_factors = None
        assert len(self._weights) == len(self._underlying_nodes)

    def __str__(self):
        return self._name

    def _weighted_sum_underlying(self, normalize=True, cache=None):
        weighted_sum = 0
        for w, u in zip(self._weights, self._underlying_nodes):
            weight = w.weight()
            if weight == 0:
                continue
            weighted_sum += weight * u.value(cache)
        if normalize:
            weighted_sum /= sum([w.weight() for w in self._weights])
        return weighted_sum

    def weight_count(self):
        return len(self._weights)

    def weights(self):
        return self._weights

    def set_weights(self, weights):
        assert len(weights) == len(self._weights)
        for w, ww in zip(self._weights, weights):
            w.set_weight(ww)

    def set_weight_derivatives(self, derivatives):
        for w, d in zip(self._weights, derivatives):
            w.set_derivative(d)

    def value(self, cache=None):
        if self in cache:
            return cache[self]
        sum_of_underlying_nodes = self._weighted_sum_underlying(cache=cache)
        value = self._activation.value(sum_of_underlying_nodes)
        cache[self] = value
        return value

    def derivative(self, cache=None):
        """
        :return: the Jacobian wrt current weights
        """
        weighted_sum = self._weighted_sum_underlying(cache=cache)
        d_activation = self._activation.derivative(weighted_sum)
        return [(d_activation * u.value(cache) if not isinstance(u, ZeroNode) else np.nan) for u in self._underlying_nodes]

    def chain_rule_factors(self, cache=None):
        """
        :return: chain rule factors of node with respect to underlying weights (aka derivatives wrt underlying nodes)
        """
        weighted_sum = self._weighted_sum_underlying(cache=cache)
        d_activation = self._activation.derivative(weighted_sum)
        factors = [d_activation * w.weight() for w in self._weights]
        self._chain_rule_factors = factors
        return factors
