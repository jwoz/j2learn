import random
from timeit import default_timer
from j2learn.regression.logistic import Logistic


class GradientDescent:
    def __init__(self, model, learning_rate):
        self._model = model
        self._objective = Logistic(model)
        self._learning_rate = learning_rate

    def sgd(self, images, labels, iterations=200):
        n = len(images)
        assert n == len(labels)
        t0 = default_timer()
        j = 0
        for i in range(iterations):
            if i % 100 == 0:
                if i > 0:
                    dt = default_timer() - t0
                    j += i + 1
                    tt = dt / i * n
                    print(f' {dt / 60:6.2f}/{tt / 60:6.2f} min')
                print(f'{i}/{n} ', end='', flush=True)
            elif i % 10 == 0:
                print('+', end='', flush=True)
            r = random.randint(0, n - 1)
            image = images[r]
            label = labels[r]
            self._model.update_data_layer(image, label)
            self._model.jacobian()
            cost, chain_rule_factor = self._objective.cost(label)
            weights = self._model.weights()
            # update weights (missing the cost function)
            for w in weights:
                derivative = w.derivative()
                assert len(derivative) == 1, 'SGD for 1D derivatives only (ie. one dim model prediction)'
                delta = self._learning_rate * chain_rule_factor * w.derivative()[0]
                self._model.set_weight(w, w.weight() - delta)
