import random

from j2learn.node.node import Node
from j2learn.layer.layer import LayerBase
from j2learn.node.weight import Weight


class Dense(LayerBase):
    def __init__(self, activation, shape, underlying_layer=None, build=True, weight=None, name=''):
        self._activation = activation
        super().__init__(shape, underlying_layer, build, weight, f'dense[{name}]')

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
            weights = [Weight(w / sum(weights)) for w in weights]
            self._nodes.append(Node(self._activation, weights, nodes, name=f'{self._name}_{n}'))
        self._built = True
