class Weight:
    def __init__(self, weight):
        self._weight = weight
        self.id = id(self)
        self._derivatives = []

    def weight(self):
        return self._weight

    def set_weight(self, weight):
        self._weight = weight

    def derivative(self):
        return self._derivatives

    def set_derivative(self, derivative):
        self._derivatives.append(derivative)
