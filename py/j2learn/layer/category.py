from j2learn.layer.layer import LayerBase
from j2learn.node.maximum import MaximumNode


class Category(LayerBase):
    def __init__(self, categories, underlying_layer=None, build=True, weight=None, name=''):
        self._categories = categories
        super().__init__((1, 1), underlying_layer, build, weight, f'category[{name}]')
        assert underlying_layer is None or len(categories) == self._underlying_layer.shape(0)
        assert underlying_layer is None or self._underlying_layer.shape(1) == 1

    def initialize(self, underlying_layer, build):
        super().initialize(underlying_layer, build)
        assert len(self._categories) == self._underlying_layer.shape(0)
        assert self._underlying_layer.shape(1) == 1

    def build(self, init=None):
        self._nodes = [MaximumNode(self._categories, [self._underlying_layer.node(i) for i in range(len(self._categories))])]
        self._built = True

    def set_weights(self, weights):
        pass

    def weights(self):
        return []

    def value(self, cache=None):
        return self._nodes[0].value(cache)

    def predict(self, index=False, cache=None):
        return self._nodes[0].predict(cache)

    def chain_rule_factors(self, upper_layer_factors=None, cache=None):
        """
        For the chain rule factors, it is assumed that the maximum node is substantially larger than the others.
        Ie. derivatives of the lower layers are zero except wrt to the node with max value.
        """
        i = self._nodes[0].node_index(cache)
        return [[(1 if j == i else 0) for j in range(self._underlying_layer.node_count())]]
