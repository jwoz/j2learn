class WeightedNode:
    def __init__(self, weight, node):
        self._node = node
        self._weight = weight
        self._id = id(self)

    def weight(self):
        return self._weight

    def value(self):
        return self._node.value()

    def node(self):
        return self._node