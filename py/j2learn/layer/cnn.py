from j2learn.node.node import Node


class CNN:
    def __init__(self, kernel, stride, underlying_layer):
        self._underlying_layer = underlying_layer
        self._kernel = kernel
        self._stride = stride
        self._nx = None
        self._ny = None

    def build(self):
        shape = self._underlying_layer.shape()
        for nx in range(shape[0]):
            for ny in range(shape[1]):
                n = nx + ny * shape[0]
                # collect nodes according to kernel and stride
                node = Node()
