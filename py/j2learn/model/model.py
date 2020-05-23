from j2learn.etc.tools import flatten as flatten_list


class Model:
    def __init__(self, layers):
        self._layers = layers
        self._weight_counts = []
        self._weights = []
        self._compiled = False
        self._built = False

    def compile(self, build=False):
        underlying_layer = None
        for layer in self._layers:
            if not layer.is_root:
                layer.initialize(underlying_layer, build)
            underlying_layer = layer
        self._compiled = True
        self._built = False
        if build:
            self.build()

    def build(self):
        assert self._compiled, 'Model must be compiled first.'
        for layer in self._layers:
            layer.build()
        self._built = True
        self._weights_counted = False

    def weight_count(self):
        if not len(self._weights):
            self.weights()
        return len(self._weights)

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
        if rescan or not len(self._weights):
            self.weights(reset=True)
        assert len(weights) == len(self._weights)
        for w, ww in zip(weights, self._weights):
            ww.set_weight(w)

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
        layer = self._layers[-1]
        jacobian = layer.jacobian()
        return jacobian
