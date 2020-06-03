from itertools import cycle

from j2learn.etc.linear_algebra import matrix_product


class LayerBase:
    is_root = False

    def __init__(self, shape: (tuple, None), underlying_layer, build: bool, weight: (float, None), name: str):
        self._underlying_layer = underlying_layer
        self._shape = shape
        self._nodes = []
        self._built = False
        self._name = name
        if build and underlying_layer is not None:
            self.build(init=weight)
        self._chain_rule_factors = []
        self._derivatives = []

    def __str__(self):
        return self._name

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

    def node_count(self):
        return len(self._nodes)

    def node(self, i, j=None):
        if j is None:
            return self._nodes[i]
        assert i * self._shape[1] + j < len(self._nodes), f'{i}, {j}, {self._shape}: {i * self._shape[1] + j} < {len(self._nodes)}'
        return self._nodes[i * self._shape[1] + j]

    def nodes(self):
        return self._nodes

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

    def predict(self, cache=None):
        return self.value(cache)

    def value(self, cache=None):
        return [node.value(cache) for node in self._nodes]

    def node_derivatives(self, cache=None):
        return [node.derivative(cache) for node in self._nodes]

    def node_chain_rule_factors(self, cache=None):
        return [node.chain_rule_factors(cache) for node in self._nodes]

    def set_weight_derivatives(self, derivatives):
        for ds, ns in zip(derivatives, cycle(self._nodes)):
            ns.set_weight_derivatives(ds)

    def jacobian(self, upper_layer_factors=None, cache=None):
        if len(self._derivatives):
            return self._derivatives
        derivatives = self.node_derivatives(cache)
        if upper_layer_factors is not None and len(upper_layer_factors):
            m = []
            for f in upper_layer_factors:
                for ff, d in zip(f, derivatives):
                    r = []
                    for dd in d:
                        r.append(dd * ff)
                    m.append(r)
            derivatives = m
        self._derivatives = derivatives
        self.set_weight_derivatives(derivatives)
        return derivatives

    def chain_rule_factors(self, upper_layer_factors=None, cache=None):
        if len(self._chain_rule_factors):
            return self._chain_rule_factors
        factors = self.node_chain_rule_factors(cache)
        if upper_layer_factors is not None and len(upper_layer_factors):
            factors = matrix_product(upper_layer_factors, factors)
        self._chain_rule_factors = factors
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
