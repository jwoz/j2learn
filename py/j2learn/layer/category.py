from j2learn.node.node import MaximumNode
import random
from j2learn.layer.layer import LayerBase

class Category(LayerBase):
    def __init__(self, categories, underlying_layer, build=True):
        self._categories = categories
        self._underlying_layer = underlying_layer
        assert len(categories) == self._underlying_layer.shape()[0]
        assert self._underlying_layer.shape()[1] == 1
        self._shape = (1, 1)
        self._node = None
        if build:
            self.build()

    def build(self):
        self._node = MaximumNode(self._categories, [self._underlying_layer.node(i) for i in range(len(self._categories))])

    def node(self, i, j=None):
        assert i == 1 and (j is None or j ==1)
        return self._node

    def shape(self):
        return self._shape
