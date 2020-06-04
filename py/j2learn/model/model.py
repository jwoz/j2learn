from j2learn.etc.tools import flatten as flatten_list
from j2learn.node.weight import ZeroWeight


class Model:
    def __init__(self, layers):
        self._layers = layers
        self._weight_counts = []
        self._weights = []
        self._compiled = False
        self._built = False
        self._node_value_cache = {}

    def compile(self, build=False):
        underlying_layer = None
        for layer in self._layers:
            if not layer.is_root:
                layer.initialize(underlying_layer, build)
            underlying_layer = layer
        self._compiled = True
        self._built = build

    def clear_cache(self):
        self._node_value_cache = {}

    def build(self):
        assert self._compiled, 'Model must be compiled first.'
        for layer in self._layers:
            layer.build()
        self._built = True

    def update_data_layer(self, data, label=None, maximum=255):
        self._layers[0].set_image_data_and_label(data, label, maximum)
        self._node_value_cache = {}

    def weight_count(self):
        if not len(self._weights):
            self.weights()
        return len(self._weights)

    def weights(self):
        weights = []
        n = 0
        for layer in self._layers:
            if layer.is_root:
                assert n == 0
                continue
            layer_weights = flatten_list(layer.weights())
            weights.append([w for w in layer_weights if not isinstance(w, ZeroWeight)])
        weights = list(flatten_list(weights))
        self._weights = weights
        return weights

    def predict(self, index=False):
        assert self._built, 'The model has not been built, cannot predict'
        return self._layers[-1].predict(index=index, cache=self._node_value_cache)

    def value(self):
        assert self._built, 'The model has not been built, cannot predict'
        return self._layers[-1].value(self._node_value_cache)

    def probability(self):
        assert self._built, 'The model has not been built, cannot predict'
        assert hasattr(self._layers[-1], 'probability'), 'Final layer does not have a probability method.'
        return self._layers[-1].probability(self._node_value_cache)

    def set_weight(self, weight, value):
        self._node_value_cache = {}
        weight.set_weight(value)
        weight.reset()

    def jacobian(self):
        factors = []
        jacobians = []
        for layer in self._layers[:0:-1]:
            jacobian = layer.jacobian(factors, self._node_value_cache)
            jacobians.insert(0, jacobian)
            factors = layer.chain_rule_factors(factors, self._node_value_cache)
        return jacobians

    def chain_rule_factors(self):
        layer_factors = []
        factors = []
        for layer in self._layers[:0:-1]:
            factors = layer.chain_rule_factors(factors, self._node_value_cache)
            layer_factors.append(factors)
        return layer_factors
