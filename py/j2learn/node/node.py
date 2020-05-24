class Node:
    def __init__(self, activation, weights, underlying_nodes, name=''):
        self._activation = activation
        self._weights = weights
        self._underlying_nodes = underlying_nodes
        self._name = name
        assert len(self._weights) == len(self._underlying_nodes)

    def __str__(self):
        return self._name

    def _weighted_sum_underlying(self, normalize=False):
        weighted_sum = sum([w.weight() * u.value() for w, u in zip(self._weights, self._underlying_nodes)])
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

    def value(self):
        sum_of_underlying_nodes = self._weighted_sum_underlying()
        return self._activation.value(sum_of_underlying_nodes)

    def derivative(self, chain_rule_factor=1):
        """
        :return: the Jacobian wrt current weights
        """
        weighted_sum = self._weighted_sum_underlying()
        d_activation = self._activation.derivative(weighted_sum)
        for w, u in zip(self._weights, self._underlying_nodes):
            w.set_derivative(chain_rule_factor * d_activation * u.value(), name=f'{self}/{u}')
        return [w.derivative() for w in self._weights]

    def chain_rule_factors(self, chain_rule_factor=1):
        """
        :return: the Jacobian wrt current weights
        """
        weighted_sum = self._weighted_sum_underlying()
        d_activation = self._activation.derivative(weighted_sum)
        factors = [chain_rule_factor * d_activation * w.weight() for w in self._weights]
        return factors
