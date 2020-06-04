import math


class Logistic:
    def __init__(self, model):
        self._model = model

    def cost(self, expected_label):
        """
        :return: value and chain rule factor
        """
        value = max(self._model.value())
        label = self._model.predict()
        index = self._model.predict(index=True)
        if value < 0 or value > 1:
            print(f'probability out of range: {value}, {expected_label} vs {self._model.predict()}')
            return None, None, None
        if label == expected_label:
            return -math.log(value), -1.0 / value, index
        return -math.log(1.0 - value), 1.0 / (1.0 - value), index
