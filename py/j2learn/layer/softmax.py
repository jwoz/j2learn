import numpy as np

from j2learn.layer.layer import LayerBase
from j2learn.node.softmax import SoftMaxNode
from j2learn.node.weight import Weight


class SoftMax(LayerBase):
    def __init__(self, categories: list, underlying_layer: (list, None) = None, build: bool = True, weight: (float, None) = None, name: str = ''):
        self._categories = categories
        super().__init__((len(categories), 1), underlying_layer, build, weight, f'softmax[{name}]')
        assert underlying_layer is None or len(categories) == self._underlying_layer.shape(0)
        assert underlying_layer is None or self._underlying_layer.shape(1) == 1

    def initialize(self, underlying_layer, build):
        super().initialize(underlying_layer, build)
        # assert len(self._categories) == self._underlying_layer.shape(0)
        assert self._underlying_layer.shape(1) == 1

    def build(self, init=None):
        # take the underlying nodes and build n sums with separate weights, where n is the number of outputs
        weight_count = len(self._categories) * self._underlying_layer.node_count()
        weights = np.random.uniform(size=weight_count) if init is None else np.full(weight_count, init)
        self._nodes = [SoftMaxNode(self._categories, [Weight(w / sum(weights)) for w in weights], self._underlying_layer.nodes())]
        self._built = True

    def set_weights(self, weights):
        pass

    def weights(self):
        return self._nodes[0].weights()

    def value(self, cache=None):
        return self._nodes[0].value(cache)

    def predict(self, cache=None):
        return self._nodes[0].predict(cache)

    def chain_rule_factors(self, upper_layer_factors=None, cache=None):
        """
        For the chain rule factors, it is assumed that the maximum node is substantially larger than the others.
        Ie. derivatives of the lower layers are zero except wrt to the node with max value.
        """
        i = self._nodes[0].node_index(cache)
        return [[(1 if j == i else 0) for j in range(self._underlying_layer.node_count())]]
