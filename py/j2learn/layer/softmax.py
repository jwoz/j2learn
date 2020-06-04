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
        assert self._underlying_layer.shape(1) == 1

    def build(self, init=None):
        # take the underlying nodes and build n sums with separate weights, where n is the number of outputs
        weight_count = len(self._categories) * self._underlying_layer.node_count()
        weights = np.random.uniform(size=weight_count) if init is None else np.full(weight_count, init)
        self._nodes = [SoftMaxNode(self._categories, [Weight(w / sum(weights), f'{self._name} [{i}]') for i, w in enumerate(weights)], self._underlying_layer.nodes())]
        self._built = True

    def set_weights(self, weights):
        i = 0
        for node in self._nodes:
            j = node.weight_count()
            node.set_weights(weights[i:i + j])
            i += j

    def weights(self):
        return self._nodes[0].weights()

    def value(self, cache=None):
        return self._nodes[0].value(cache)

    def predict(self, index=False, cache=None):
        return self._nodes[0].predict(index, cache)

    def node_derivatives(self, cache=None):
        return self._nodes[0].derivative(cache)

    def node_chain_rule_factors(self, cache=None):
        return self._nodes[0].chain_rule_factors(cache)

    def set_weight_derivatives(self, derivatives):
        self._nodes[0].set_weight_derivatives(derivatives)
