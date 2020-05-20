class LayerBase:
    def __init__(self, shape, underlying_layer, build, weight):
        self._underlying_layer = underlying_layer
        self._shape = shape
        self._nodes = []
        self._built = False
        if build and underlying_layer is not None:
            self.build(init=weight)

    def initialize(self, underlying_layer, build):
        self._underlying_layer = underlying_layer
        if build:
            self.build()

    def build(self, init=None):
        raise NotImplementedError('Child class must implement')

    def count(self):
        if not self._built:
            print('Layer not built, no weights to count')
            return 0
        n = 0
        for node in self._nodes:
            n += len(node.weights())
        return n

    def node(self, i, j=None):
        if j is None:
            return self._nodes[i]
        assert i * self._shape[1] + j < len(self._nodes), f'{i}, {j}, {self._shape}: {i * self._shape[1] + j} < {len(self._nodes)}'
        return self._nodes[i * self._shape[1] + j]

    def shape(self, dimension=None):
        if dimension is None:
            return self._shape
        assert dimension < len(self._shape)
        return self._shape[dimension]

    def weights(self):
        return [node.weights() for node in self._nodes]

    def value(self):
        return [node.value() for node in self._nodes]

    def jacobian(self):
        partial_derivatives = [node.derivative() for node in self._nodes]
        return partial_derivatives

    def display(self, numbers=False, threshold=0.8):
        render = ''
        for i in range(len(self._nodes)):
            if i % self._shape[0] == 0:
                render += '\n'
            if numbers:
                render += f'{self._nodes[i].value():4.2f} ' if self._nodes[i].value() > 0.01 else '.... '
            elif self._nodes[i].value() > threshold:
                render += '@'
            else:
                render += '.'
        return render
