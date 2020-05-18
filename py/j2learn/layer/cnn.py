from j2learn.node.node import Node
from j2learn.node.data import ZeroNode
import random

class CNN:
    def __init__(self, activation, kernel, stride, underlying_layer, build=True):
        self._activation = activation
        self._underlying_layer = underlying_layer
        self._kernel = kernel
        self._stride = stride
        self._nodes = []
        if build:
            self.build()

    def build(self):
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
                        kk = nx - self._kernel[0] // 2 + k + k*self._stride[0]
                        ll = ny - self._kernel[1] // 2 + l + l * self._stride[1]
                        if kk < 0 or ll < 0 or kk >= shape[0] or ll >= shape[1]:
                            node = ZeroNode()
                            weight = 0
                        else:
                            node = self._underlying_layer.node(kk, ll)
                            weight = random.random()
                        nodes.append(node)
                        weights.append(weight)
                this_cnn_node = Node(self._activation, weights, nodes)
                self._nodes.append(this_cnn_node)
        self._shape = shape

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
