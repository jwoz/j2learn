import random

from j2learn.node.node import Node


class Dense:
    is_root = False

    def __init__(self, activation, shape, underlying_layer=None, build=True):
        self._activation = activation
        self._underlying_layer = underlying_layer
        self._shape = shape
        self._nodes = []
        self._built = False
        if underlying_layer is None:
            return
        if build:
            self.build()

    def initialize(self, underlying_layer, build):
        self._underlying_layer = underlying_layer
        if build:
            self.build()

    def build(self):
        shape = self._underlying_layer.shape()
        self._nodes = []
        for n in range(self._shape[0] * self._shape[1]):
            nodes = []
            weights = []
            for m in range(shape[0] * shape[1]):
                node = self._underlying_layer.node(m)
                weight = random.random()
                nodes.append(node)
                weights.append(weight)
            weights = [w / sum(weights) for w in weights]
            self._nodes.append(Node(self._activation, weights, nodes))
        self._built = True

    def count(self):
        if not self._built:
            print('Layer not built, no weights count')
            return 0
        n = 0
        for node in self._nodes:
            n += len(node.weights())
        return n


    def node(self, i, j=None):
        if j is None:
            return self._nodes[i]
        assert i * self._shape[1] + j < len(
            self._nodes), f'{i}, {j}, {self._shape}: {i * self._shape[1] * j} < {len(self._nodes)}'
        return self._nodes[i * self._shape[1] * j]

    def shape(self):
        return self._shape

    def jacobian(self):
        partial_derivatives = [node.derivative() for node in self._nodes]
        return partial_derivatives
