from j2learn.etc.tools import flatten as flatten_list


class Model:
    def __init__(self, layers):
        self._layers = layers
        self._weight_counts = []
        self._weights = []
        self._compiled = False
        self._built = False
        self._weights_counted = False

    def compile(self, build=False):
        underlying_layer = None
        for layer in self._layers:
            if not layer.is_root:
                layer.initialize(underlying_layer, build)
            underlying_layer = layer
        self._compiled = True
        self._built = False
        self._weights_counted = False
        if build:
            self.build()

    def build(self):
        assert self._compiled, 'Model must be compiled first.'
        for layer in self._layers:
            layer.build()
        self._built = True
        self._weights_counted = False

    def weight_count(self):
        if not self._weights_counted:
            self.weight_counts()
        return self._weight_counts[-1][1]

    def weight_counts(self):
        if not self._weights_counted:
            counts = []
            n = 0
            for layer in self._layers:
                layer_count = layer.weight_count()
                counts.append((n, layer_count + n))
                n += layer_count
            self._weight_counts = counts
            self._weights_counted = True
        return self._weight_counts

    def weights(self, flatten=True, reset=True):
        weights = []
        n = 0
        for layer in self._layers:
            if layer.is_root:
                assert n == 0
                continue
            layer_weights = layer.weights()
            weights.append(layer_weights)
        if reset:
            self._weights = weights
        return list(flatten_list(weights)) if flatten else weights

    def set_weights(self, weights, rescan=False):
        if not self._built:
            print('Model must be built for each new image and at least once.')
        n = len(weights)
        if rescan or not self._weights_counted:
            self.weight_counts()
        assert n == self.weight_count()
        for count, layer in zip(self._weight_counts, self._layers):
            layer.set_weights(weights[count[0]:count[1]])

    def predict(self):
        assert self._built, 'The model has not been built, cannot predict'
        return self._layers[-1].value()

    def value(self):
        assert self._built, 'The model has not been built, cannot predict'
        return self._layers[-1].value()

    def probability(self):
        assert self._built, 'The model has not been built, cannot predict'
        assert hasattr(self._layers[-1], 'probability'), 'Final layer does not have a probability method.'
        return self._layers[-1].probability()

    def jacobian(self):
        jacobians = []
        chain_rule_factors = []
        for layer in self._layers[::-1]:
            partial_jacobian = layer.jacobian()
            jacobian = []
            if len(chain_rule_factors):
                for f in chain_rule_factors:
                    for ff, j in zip(f, partial_jacobian):
                        derivatives = [ff * jj for jj in j]
                        jacobian.append(derivatives)
            else:
                jacobian = partial_jacobian
            jacobians.append(jacobian)
            chain_rule_factors = layer.chain_rule_factors()
        return jacobians
