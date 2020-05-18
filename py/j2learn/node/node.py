class Node:
    def __init__(self, activation, weights, underlying_nodes):
        self._activation = activation
        self._weights = weights
        self._underlying_nodes = underlying_nodes

    def _sum(self):
        return sum([w * u.value() for w, u in zip(self._weights, self._underlying_nodes)])

    def value(self):
        sum_of_underlying_nodes = self._sum()
        return self._activation.value(sum_of_underlying_nodes)

    def derivative(self):
        """
        :return: the Jacobian wrt current weights
        """
        weighted_sum = self._sum()
        d_activation = self._activation.derivative(weighted_sum)
        this_derivative = [d_activation * u.value() for u in self._underlying_nodes]
        return this_derivative

    def update_weights(self, new_weights):
        self._weights = new_weights
