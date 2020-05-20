class Node:
    def __init__(self, activation, weights, underlying_nodes):
        self._activation = activation
        self._weights = weights
        self._underlying_nodes = underlying_nodes

    def _weighted_sum_underlying(self):
        weighted_sum = sum([w * u.value() for w, u in zip(self._weights, self._underlying_nodes)])
        return weighted_sum / sum(self._weights)

    def weight_count(self):
        assert len(self._weights) == len(self._underlying_nodes)
        return len(self._weights)

    def weights(self):
        return self._weights

    def set_weights(self, weights):
        self._weights = weights

    def value(self):
        sum_of_underlying_nodes = self._weighted_sum_underlying()
        return self._activation.value(sum_of_underlying_nodes)

    def derivative(self):
        """
        :return: the Jacobian wrt current weights
        """
        weighted_sum = self._weighted_sum_underlying()
        d_activation = self._activation.derivative(weighted_sum)
        this_derivative = [d_activation * u.value() for u in self._underlying_nodes]
        return this_derivative

    def update_weights(self, new_weights):
        self._weights = new_weights
