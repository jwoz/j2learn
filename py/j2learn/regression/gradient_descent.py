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
                    print(f' *** {self._model.value()[0]:6.3f}, {self._model.predict()[0]}, {self._model._layers[0].label()}: {self._objective.cost(self._model._layers[0].label())[0]:6.3g},  {self._objective.cost(self._model._layers[0].label())[1]:6.3g} ***')
                print(f'{i}/{iterations} ', end='', flush=True)
            elif i % 10 == 0:
                print('+', end='', flush=True)
            self._model.update_data_layer(image, label)
            self._model.jacobian()
            cost, chain_rule_factor = self._objective.cost(label)
            if cost is None:
                continue
            i += 1
            weights = self._model.weights()
            sum_delta = 0
            for w in weights:
                derivative = w.derivative()
                assert len(derivative) == 1, 'SGD for 1D derivatives only (ie. one dim model prediction)'
                delta = self._learning_rate * chain_rule_factor * w.derivative()[0]
                sum_delta -= delta

                if i % int(iterations / 100) == 0 and i > 0:
                    snapshot.append([i, w.id, w.name, w.weight(), delta])
                self._model.set_weight(w, w.weight() - delta)

            if i % int(iterations / 100) == 0 and i > 0:
                pd.DataFrame(data=snapshot, columns=['iteration', 'id', 'name', 'weight', 'delta']).to_csv('five_snapshot.csv')
