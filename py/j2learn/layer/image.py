import math

from j2learn.layer.layer import LayerBase
from j2learn.node.node import DataNode


class Image(LayerBase):
    def __init__(self, image_data, shape=None, label=None):
        self._label = label

        if shape is None:
            shape = (int(math.sqrt(len(image_data))),) * 2
        assert shape[0] * shape[1] == len(image_data)
        self._shape = shape
        self._nodes = [DataNode(x) for x in image_data]

    def label(self):
        return self._label

    def value(self, x, y):
        assert x < self._shape[0] and y < self._shape[1]
        return self._nodes[x + self._shape[0] * y].value()

    def node(self, i):
        ...

    def shape(self):
        return self._shape

    def display(self, threshold=200):
        render = ''
        for i in range(len(self._nodes)):
            if i % self._shape[0] == 0:
                render += '\n'
            if self._nodes[i].value() > threshold:
                render += '@'
            else:
                render += '.'
        return render
