from abc import ABC, abstractmethod

class NodeBase:
    @abstractmethod
    def value(self):
        ...

    @abstractmethod
    def derivative(self):
        ...


class Node(NodeBase):
    def __init__(self, activation, weights, underlying_nodes):
        self._activation = activation
        self._weights = weights
        self._underlying_nodes = underlying_nodes

    def value(self):
        sum_of_underlying_nodes = sum([w * u.value() for w, u in zip(self._weights, self._underlying_nodes)])
        return self._activation.value(sum_of_underlying_nodes)

    def derivative(self):
        """
        :return: the Jacobian wrt current weights
        """
        ...

    def update_weights(self, new_weights):
        self._weights = new_weights


class DataNode(NodeBase):
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def derivative(self):
        return 0

