import random

from j2learn.layer.layer import LayerBase
from j2learn.node.node import Node
from j2learn.node.weight import Weight


class CNN(LayerBase):
    def __init__(self, activation, kernel, stride=None, underlying_layer=None, build=True, weight=None, name=''):
        self._activation = activation
        self._kernel = kernel
        self._stride = (0, 0) if stride is None else stride
        super().__init__(None, underlying_layer, build, weight, f'cnn[{name}]')

    def build(self, init=None):
        shape = self._underlying_layer.shape()
        self._nodes = []
        for nx in range(shape[0]):
            for ny in range(shape[1]):
                nodes = []
                weights = []
                # collect nodes according to kernel and stride
                for k in range(self._kernel[0]):
                    for l in range(self._kernel[1]):
                        kk = nx - self._kernel[0] // 2 + k + k * self._stride[0]
                        ll = ny - self._kernel[1] // 2 + l + l * self._stride[1]
                        if not (kk < 0 or ll < 0 or kk >= shape[0] or ll >= shape[1]):
                            node = self._underlying_layer.node(kk, ll)
                            weight = random.random() if init is None else init
                            nodes.append(node)
                            weights.append(weight)
                weights = [Weight(w / sum(weights)) for w in weights]
                this_cnn_node = Node(self._activation, weights, nodes)
                self._nodes.append(this_cnn_node)
        self._shape = shape
        self._built = True
