import random

from j2learn.layer.layer import LayerBase
from j2learn.node.node import Node
from j2learn.node.weight import Weight


class Dense(LayerBase):
    def __init__(self, activation, shape, underlying_layer=None, build=True, weight=None, name=''):
        self._activation = activation
        super().__init__(shape, underlying_layer, build, weight, f'dense[{name}]')

    def build(self, init=None):
        if self._built:
            raise AttributeError('Layer already built.')
        shape = self._underlying_layer.shape()
        self._nodes = []
        for n in range(self._shape[0] * self._shape[1]):
            underlying_nodes = []
            weights = []
            for m in range(shape[0] * shape[1]):
                underlying_nodes.append(self._underlying_layer.node(m))
                weights.append(random.random() if init is None else init)
            weights = [Weight(w / sum(weights), name=f'{self._name} [{n}]: : {m}') for m, w in enumerate(weights)]
            self._nodes.append(Node(self._activation, weights, underlying_nodes, name=f'{self._name}_{n}'))
        self._built = True
