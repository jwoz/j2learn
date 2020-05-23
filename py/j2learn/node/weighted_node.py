class WeightedNode:
    def __init__(self, weight, node):
        self._node = node
        self._weight = weight
        self._id = id(self)

        self._value = None
        self._derivative = None

    def weight(self):
        return self._weight

    def set_weight(self, weight):
        self._weight = weight

    def value(self):
        return self._node.value()

    def node(self):
        return self._node

    def derivative(self):
        return self._derivative

    def set_derivative(self, derivative):
        self._derivative = derivative
