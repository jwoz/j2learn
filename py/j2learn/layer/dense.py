import random

from j2learn.node.node import Node
from j2learn.layer.layer import LayerBase
from j2learn.node.weighted_node import WeightedNode


class Dense(LayerBase):
    def __init__(self, activation, shape, underlying_layer=None, build=True, weight=None):
        self._activation = activation
        super().__init__(shape, underlying_layer, build, weight)

    def build(self, init=None):
        shape = self._underlying_layer.shape()
        self._nodes = []
        for n in range(self._shape[0] * self._shape[1]):
            nodes = []
            weights = []
            for m in range(shape[0] * shape[1]):
                node = self._underlying_layer.node(m)
                weight = random.random() if init is None else init
                nodes.append(node)
                weights.append(weight)
            weights = [w / sum(weights) for w in weights]
            self._nodes.append(Node(self._activation, [WeightedNode(weight, node) for weight, node in zip(weights, nodes)]))
        self._built = True
