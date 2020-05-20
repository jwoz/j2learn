import math


class Logistic:
    def __init__(self, model, cost_function):
        self._model = model
        self._cost_function = cost_function

    def cost(self, expected_label):
        if self._model.predict() == expected_label:
            return math.log(self._model.probability())
        return math.log(1 - self._model.predict())
