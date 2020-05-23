class Weight:
    def __init__(self, weight):
        self._weight = weight
        self.id = id(self)
        self._derivative = []

    def weight(self):
        return self._weight

    def set_weight(self, weight):
        self._weight = weight

    def derivative(self):
        return self._derivative

    def set_derivative(self, derivative):
        self._derivative.append(derivative)
