import numpy as np
import random
from timeit import default_timer

import pandas as pd

from j2learn.regression.logistic import Logistic


class GradientDescent:
    def __init__(self, model, learning_rate, labels=None):
        self._model = model
        self._objective = Logistic(model)
        self._learning_rate = learning_rate
        self._labels = labels  # TODO not in the right place

    def sgd(self, images, labels, iterations=200):
        n = len(images)
        assert n == len(labels)
        t0 = default_timer()
        j = 0
        sum_delta = None
        i = 0
        snapshot = []
        while i < iterations:
            r = random.randint(0, n - 1)
            image = images[r]
            label = labels[r]
            if self._labels is not None and label not in self._labels:
                continue
            if i % 100 == 0:
                if i > 0:
                    dt = default_timer() - t0
                    j += i + 1
                    tt = dt / i * iterations
                    print(f' {dt / 60:6.2f}/{tt / 60:6.2f} min, sum(delta)={sum_delta:8.5g}', end='\t')
                    probabilities = ', '.join([f'{v:5.2f}' for v in self._model.value()])
                    print(f' *** {max(self._model.value()):6.3f}, {self._model.predict()}, {self._model._layers[0].label()} ## [{probabilities}] ***')
                print(f'{i}/{iterations} ', end='', flush=True)
            elif i % 10 == 0:
                print('+', end='', flush=True)
            self._model.update_data_layer(image, label)
            self._model.jacobian()
            costs, chain_rule_factors = self._objective.cost(label)
            i += 1
            weights = self._model.weights()
            sum_delta = 0
            for w in weights:
                derivatives = w.derivative()
                derivative = sum([crf*d for crf, d in zip(chain_rule_factors, derivatives) if not np.isnan(d*crf)])
                delta = self._learning_rate * derivative
                sum_delta += abs(derivative)

                if i % int(iterations / 100) == 0 and i > 0:
                    snapshot.append([i, w.id, w.name, w.weight(), delta])
                self._model.set_weight(w, w.weight() - delta)

            if i % int(iterations / 100) == 0 and i > 0:
                pd.DataFrame(data=snapshot, columns=['iteration', 'id', 'name', 'weight', 'delta']).to_csv('sgd_snapshot.csv')
