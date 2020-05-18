from j2learn.node.node import Node
import random

class Dense:
    def __init__(self, activation, shape, underlying_layer, build=True):
        self._activation = activation
        self._underlying_layer = underlying_layer
        self._shape = shape
        self._nodes = []
        if build:
            self.build()

    def build(self):
        shape = self._underlying_layer.shape()
        self._nodes = []
        for n in range(self._shape[0] * self._shape[1]):
            nodes = []
            weights = []
            for m in range(shape[0]*shape[1]):
                node = self._underlying_layer.node(m)
                weight = random.random()
                nodes.append(node)
                weights.append(weight)
            self._nodes.append(Node(self._activation, weights, nodes))

    def node(self, i, j=None):
        if j is None:
            return self._nodes[i]
        assert i+self._shape[0]*j < len(self._nodes), f'{i}, {j}, {self._shape}: {i+self._shape[0]*j} < {len(self._nodes)}'
        return self._nodes[i+self._shape[0]*j]

    def shape(self):
        return self._shape

    def jacobian(self):
        partial_derivatives = [node.derivative() for node in self._nodes]
        return partial_derivatives
