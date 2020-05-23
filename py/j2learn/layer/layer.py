class LayerBase:
    is_root = False

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

    def weight_count(self):
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

    def set_weights(self, weights):
        i = 0
        for node in self._nodes:
            j = node.weight_count()
            node.set_weights(weights[i:i + j])
            i += j

    def weights(self):
        return [node.weights() for node in self._nodes]

    def value(self):
        return [node.value() for node in self._nodes]

    def jacobian(self, chain_rule_factors=None):
        derivatives = []
        if chain_rule_factors is None or not len(chain_rule_factors):
            partial_derivatives = [node.derivative() for node in self._nodes]
        else:
            partial_derivatives = [node.derivative(f) for f, node in zip(chain_rule_factors, self._nodes)]
        derivatives.insert(0, partial_derivatives)
        chain_rule_factors = self.chain_rule_factors(chain_rule_factors)
        for factors in chain_rule_factors:
            underlying_derivatives = self._underlying_layer.jacobian(factors)
            if len(underlying_derivatives):
                derivatives.insert(0, underlying_derivatives)
        return derivatives

    def jacobian_broken(self, chain_rule_factors=None):
        if chain_rule_factors is None or not len(chain_rule_factors):
            partial_derivatives = [node.derivative() for node in self._nodes]
        else:
            partial_derivatives = [[node.derivative(f) for f, node in zip(crf, self._nodes)] for crf in chain_rule_factors]
        return partial_derivatives

    def chain_rule_factors(self, chain_rule_factors=None):
        if chain_rule_factors is None or not len(chain_rule_factors):
            factors = [node.chain_rule_factors() for node in self._nodes]
        else:
            factors = [node.chain_rule_factors(f) for f, node in zip(chain_rule_factors, self._nodes)] # returns 2d array: list of nodes of list of underlying nodes
        return factors

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
