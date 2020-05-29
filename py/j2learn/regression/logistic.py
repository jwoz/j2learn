import math


class Logistic:
    def __init__(self, model):
        self._model = model

    def cost(self, expected_label):
        """
        :return: value and chain rule factor
        """
        value = self._model.value()
        if self._model.predict() == [expected_label]:
            return math.log(value), 1.0/value
        return math.log(1.0 - value), -1.0/value
