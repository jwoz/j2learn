import random

from j2learn.node.data import ZeroNode
from j2learn.node.node import Node


class CNN:
    is_root = False

    def __init__(self, activation, kernel, stride, underlying_layer=None, build=True, weight=None):
        self._activation = activation
        self._kernel = kernel
        self._stride = stride
        self._shape = None
        self._nodes = []
        self._underlying_layer = underlying_layer
        self._built = False
        if underlying_layer is None:
            return
        if build:
            self.build(init=weight)

    def initialize(self, underlying_layer, build):
        self._underlying_layer = underlying_layer
        if build:
            self.build()

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

    def node(self, i, j=None):
        if j is None:
            return self._nodes[i]
        assert i * self._shape[1] + j < len(self._nodes), f'{i}, {j}, {self._shape}: {i * self._shape[1] + j} < {len(self._nodes)}'
        return self._nodes[i * self._shape[1] + j]

    def count(self):
        if not self._built:
            print('Layer not built, no weights count')
            return 0
        n = 0
        for node in self._nodes:
            n += len(node.weights())
        return n

    def shape(self):
        return self._shape

    def value(self):
        return [node.value() for node in self._nodes]

    def jacobian(self):
        partial_derivatives = [node.derivative() for node in self._nodes]
        return partial_derivatives

    def display(self, numbers=False, threshold=0.8):
        render = ''
        for i in range(len(self._nodes)):
            if i % self._shape[0] == 0:
                render += '\n'
            if numbers:
                render += f'{self._nodes[i].value():4.2f} ' if self._nodes[i].value() > 0.01 else '.... '
            elif self._nodes[i].value() > threshold:
                render += '@'
            else:
                render += '.'
        return render
