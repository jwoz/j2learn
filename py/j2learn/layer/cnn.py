import random

from j2learn.node.data import ZeroNode
from j2learn.node.node import Node
from j2learn.layer.layer import LayerBase


class CNN(LayerBase):
    is_root = False

    def __init__(self, activation, kernel, stride, underlying_layer=None, build=True, weight=None):
        self._activation = activation
        self._kernel = kernel
        self._stride = stride
        super().__init__(None, underlying_layer, build, weight)

    def build(self, init=None):
        shape = self._underlying_layer.shape()
        self._nodes = []
        for nx in range(shape[0]):
            for ny in range(shape[1]):
                n = nx + ny * shape[0]
                nodes = []
                weights = []
                # collect nodes according to kernel and stride
                for k in range(self._kernel[0]):
                    for l in range(self._kernel[1]):
                        kk = nx - self._kernel[0] // 2 + k + k * self._stride[0]
                        ll = ny - self._kernel[1] // 2 + l + l * self._stride[1]
                        if kk < 0 or ll < 0 or kk >= shape[0] or ll >= shape[1]:
                            node = ZeroNode()
                            weight = 0
                        else:
                            # print(f'{nx} {ny}: {kk} {ll}')
                            node = self._underlying_layer.node(kk, ll)
                            weight = random.random() if init is None else init
                        nodes.append(node)
                        weights.append(weight)
                weights = [w / sum(weights) for w in weights]
                this_cnn_node = Node(self._activation, weights, nodes)
                self._nodes.append(this_cnn_node)
        self._shape = shape
        self._built = True

