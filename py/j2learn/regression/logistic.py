import math


class Logistic:
    def __init__(self, model):
        self._model = model

    def cost(self, expected_label):
        """
        :return: value and chain rule factor
        """
        values = self._model.value()
        labels = self._model._layers[-1]._categories
        likelihoods = []
        factors = []
        m = len(labels)
        for value, label in zip(values, labels):
            if value < 0 or value > 1:
                print(f'probability out of range: {value}, {expected_label} vs {self._model.predict()}')
                likelihoods.append(0)
                factors.append(0)
            elif label == expected_label:
                likelihoods.append(-math.log(value) / m)
                factors.append(-1.0 / value / m)
            else:
                likelihoods.append(-math.log(1.0 - value) / m)
                factors.append(1.0 / (1.0 - value) / m)
        return likelihoods, factors
