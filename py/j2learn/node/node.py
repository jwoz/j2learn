class Node:
    def __init__(self, activation, weights, underlying_nodes):
        self._activation = activation
        self._weights = weights
        self._underlying_nodes = underlying_nodes

    def value(self):
        sum_of_underlying_nodes = sum([w * u.value() for w, u in zip(self._weights, self._underlying_nodes)])
        return self._activation.value(sum_of_underlying_nodes)

    def derivative(self):
        """
        the jacobian wrt current weights
        :return:
        """
        ...

    def update_weights(self, new_weights):
        self._weights = new_weights
