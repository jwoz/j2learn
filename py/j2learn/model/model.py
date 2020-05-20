class Model:
    def __init__(self, layers):
        self._layers = layers
        self._counts = []
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

    def weight_counts(self, reset=True):
        counts = []
        n = 0
        for layer in self._layers:
            layer_count = layer.weight_count()
            counts.append((n, layer_count + n))
            n += layer_count
        if reset:
            self._counts = counts
        return counts

    def weights(self, reset=True):
        weights = []
        counts = []
        n = 0
        for layer in self._layers:
            if layer.is_root:
                counts.append((0, 0))
                n = 0
                continue
            layer_weights = layer.weights()
            layer_count = len(layer_weights)
            weights.append(layer_weights)
            counts.append((n, layer_count + n))
            n += layer_count
        if reset:
            self._counts = counts
            self._weights = weights
        return weights

    def set_weights(self, weights):
        if not self._built:
            print('Model must be built for each new image and at least once.')
        n = len(weights)
        current_weights = self.weights()

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
