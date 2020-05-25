import random

from j2learn.node.node import Node
from j2learn.layer.layer import LayerBase
from j2learn.node.weight import Weight
from j2learn.etc.linear_algebra import matrix_product


class Dense(LayerBase):
    def __init__(self, activation, shape, underlying_layer=None, build=True, weight=None, name=''):
        self._activation = activation
        super().__init__(shape, underlying_layer, build, weight, f'dense[{name}]')

    def build(self, init=None):
        shape = self._underlying_layer.shape()
        self._nodes = []
        for n in range(self._shape[0] * self._shape[1]):
            underlying_nodes = []
            weights = []
            for m in range(shape[0] * shape[1]):
                underlying_nodes.append(self._underlying_layer.node(m))
                weights.append(random.random() if init is None else init)
            weights = [Weight(w / sum(weights)) for w in weights]
            self._nodes.append(Node(self._activation, weights, underlying_nodes, name=f'{self._name}_{n}'))
        self._built = True

    def jacobian(self, upper_layer_factors=None):
        if len(self._derivatives):
            return self._derivatives
        derivatives = [node.derivative() for node in self._nodes]
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
        for ds, ns in zip(derivatives, self._nodes):
            ns.set_weight_derivatives(ds)
        return derivatives

    def chain_rule_factors(self, upper_layer_factors=None):
        if len(self._chain_rule_factors):
            return self._chain_rule_factors
        factors = [node.chain_rule_factors() for node in self._nodes]
        if upper_layer_factors is not None and len(upper_layer_factors):
            factors = matrix_product(upper_layer_factors, factors)
        self._chain_rule_factors = factors
        return factors
