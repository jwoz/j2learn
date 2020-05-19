from j2learn.node.maximum import MaximumNode


class Category:
    is_root = False

    def __init__(self, categories, underlying_layer=None, build=True):
        self._categories = categories
        self._underlying_layer = underlying_layer
        self._shape = (1, 1)
        self._node = None
        if underlying_layer is None:
            return
        assert len(categories) == self._underlying_layer.shape()[0]
        assert self._underlying_layer.shape()[1] == 1
        if build:
            self.build()

    def initialize(self, underlying_layer, build):
        self._underlying_layer = underlying_layer
        assert len(self._categories) == self._underlying_layer.shape()[0]
        assert self._underlying_layer.shape()[1] == 1
        if build:
            self.build()

    def build(self):
        self._node = MaximumNode(self._categories,
                                 [self._underlying_layer.node(i) for i in range(len(self._categories))])

    def node(self, i, j=None):
        assert i == 1 and (j is None or j == 1)
        return self._node

    def shape(self):
        return self._shape

    def jacobian(self):
        partial_derivatives = [self._node.derivative()]
        return partial_derivatives
