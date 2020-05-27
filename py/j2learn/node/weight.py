class Weight:
    def __init__(self, weight, name=''):
        self._weight = weight
        self.id = id(self)
        self._derivatives = []
        self._derivative_name = None
        self.name = name

    def __str__(self):
        return self.name

    def weight(self):
        return self._weight

    def set_weight(self, weight):
        self._weight = weight

    def derivative(self):
        return self._derivatives

    def set_derivative(self, derivative, name=''):
        self._derivatives.append(derivative)
        self._derivative_name = name


class ZeroWeight:
    @staticmethod
    def weight():
        return 0

    def set_derivative(self, derivative, name=''):
        pass
