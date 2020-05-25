class Weight:
    def __init__(self, weight):
        self._weight = weight
        self.id = id(self)
        self._derivative = None
        self._name = None

    def weight(self):
        return self._weight

    def set_weight(self, weight):
        self._weight = weight

    def derivative(self):
        return self._derivative

    def set_derivative(self, derivative, name=''):
        assert self._derivative is None
        self._derivative = derivative
        self._name = name

# class Weight:
#     def __init__(self, weight):
#         self._weight = weight
#         self.id = id(self)
#         self._derivatives = []
#         self._derivative_names = []
#
#     def weight(self):
#         return self._weight
#
#     def set_weight(self, weight):
#         self._weight = weight
#
#     def derivative(self):
#         return self._derivatives
#
#     def set_derivative(self, derivative, name=''):
#         self._derivatives.append(derivative)
#         self._derivative_names.append(name)
