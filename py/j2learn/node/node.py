from j2learn.node.weighted_node import WeightedNode

class Node:
    def __init__(self, activation, weighted_underlying_nodes):
        self._activation = activation
        self._weighted_underlying_nodes = weighted_underlying_nodes

    def _weighted_sum_underlying(self, normalize=False):
        weighted_sum = sum([u.weight() * u.value() for u in self._weighted_underlying_nodes])
        if normalize:
            weighted_sum /= sum(self._weights)
        return weighted_sum

    def weight_count(self):
        return len(self._weighted_underlying_nodes)

    def weights(self):
        return [u.weight() for u in self._weighted_underlying_nodes]

    def set_weights(self, weights):
        for w, u in zip(weights, self._weighted_underlying_nodes):
            u.set_weight(w)

    def value(self):
        sum_of_underlying_nodes = self._weighted_sum_underlying()
        return self._activation.value(sum_of_underlying_nodes)

    def derivative(self, chain_rule_factor=1):
        """
        :return: the Jacobian wrt current weights
        """
        weighted_sum = self._weighted_sum_underlying()
        d_activation = self._activation.derivative(weighted_sum)
        this_derivative = [chain_rule_factor * d_activation * u.value() for u in self._weighted_underlying_nodes]
        return this_derivative

    def chain_rule_factors(self, chain_rule_factor=1):
        """
        :return: the Jacobian wrt current weights
        """
        weighted_sum = self._weighted_sum_underlying()
        d_activation = self._activation.derivative(weighted_sum)
        factors = [chain_rule_factor * d_activation * u.weight() for u in self._weighted_underlying_nodes]
        return factors


# class Node:
#     def __init__(self, activation, weights, underlying_nodes):
#         self._activation = activation
#         self._weights = weights
#         self._underlying_nodes = underlying_nodes
#
#     def _weighted_sum_underlying(self, normalize=False):
#         weighted_sum = sum([w * u.value() for w, u in zip(self._weights, self._underlying_nodes)])
#         if normalize:
#             weighted_sum /= sum(self._weights)
#         return weighted_sum
#
#     def weight_count(self):
#         assert len(self._weights) == len(self._underlying_nodes)
#         return len(self._weights)
#
#     def weights(self):
#         return self._weights
#
#     def set_weights(self, weights):
#         self._weights = weights
#
#     def value(self):
#         sum_of_underlying_nodes = self._weighted_sum_underlying()
#         return self._activation.value(sum_of_underlying_nodes)
#
#     def derivative(self, chain_rule_factor=1):
#         """
#         :return: the Jacobian wrt current weights
#         """
#         weighted_sum = self._weighted_sum_underlying()
#         d_activation = self._activation.derivative(weighted_sum)
#         this_derivative = [chain_rule_factor * d_activation * u.value() for u in self._underlying_nodes]
#         return this_derivative
#
#     def chain_rule_factors(self, chain_rule_factor=1):
#         """
#         :return: the Jacobian wrt current weights
#         """
#         weighted_sum = self._weighted_sum_underlying()
#         d_activation = self._activation.derivative(weighted_sum)
#         factors = [chain_rule_factor * d_activation * w for w in self._weights]
#         return factors
#
#     def update_weights(self, new_weights):
#         self._weights = new_weights
