import random

import numpy as np

from j2learn.layer.layer import LayerBase
from j2learn.node.data import ZeroNode
from j2learn.node.node import Node
from j2learn.node.weight import Weight, ZeroWeight


class CNN(LayerBase):
    def __init__(self, activation, kernel, stride=None, underlying_layer=None, build=True, weight=None, name=''):
        self._activation = activation
        self._kernel = kernel
        self._stride = (0, 0) if stride is None else stride
        super().__init__(None, underlying_layer, build, weight, f'cnn[{name}]')

    def build(self, init=None):
        if self._built:
            raise AttributeError('Layer already built.')
        shape = self._underlying_layer.shape()
        self._nodes = []
        for nx in range(shape[0]):
            for ny in range(shape[1]):
                nodes = []
                weights = []
                weights_sum = 0
                # collect indices according to kernel and stride
                indices = []
                for k in range(self._kernel[0]):
                    for l in range(self._kernel[1]):
                        kk = nx - self._kernel[0] // 2 + k + k * self._stride[0]
                        ll = ny - self._kernel[1] // 2 + l + l * self._stride[1]
                        if 0 <= kk < shape[0] and 0 <= ll < shape[1]:
                            indices.append((kk, ll))
                for k in range(shape[0]):
                    for l in range(shape[1]):
                        if (k, l) in indices:
                            node = self._underlying_layer.node(k, l)
                            weight = random.random() if init is None else init
                            nodes.append(node)
                            weights.append(weight)
                            weights_sum += weight
                        else:
                            nodes.append(ZeroNode())
                            weights.append(np.nan)
                weights = [Weight(w / weights_sum, name=f'{self._name} [{nx},{ny}]: {m}') if not np.isnan(w) else ZeroWeight() for m, w in enumerate(weights)]
                print(f'Building {self._name} [{nx}, {ny}]')
                this_cnn_node = Node(self._activation, weights, nodes, name=f'{self._name} [{nx},{ny}]')
                self._nodes.append(this_cnn_node)
        self._shape = shape
        self._built = True
