import numpy as np
from dataclasses import dataclass
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


class MaximumNode(NodeBase):
    def __init__(self, categories, underlying_nodes):
        self._categories = categories
        self._underlying_nodes = underlying_nodes

    def value(self):
        values = [node.value() for node in self._underlying_nodes]
        i = np.argmax(np.array(values))
        if values[i] == 0:
            return -1
        return self._categories[i]


@dataclass
class ValueNode:
    _value: float
    _derivative: float = 0
    def value(self):
        return self._value
    def derivative(self):
        return self._derivative

DataNode = ValueNode
ConstantNode = ValueNode

class ZeroNode(ValueNode):
    def __init__(self):
        super().__init__(0)