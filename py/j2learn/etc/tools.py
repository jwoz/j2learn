from collections import Iterable


def flatten(items):
    # also flatten in pandas ... import seems slow
    # from pandas.core.common import flatten
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def finite_difference(model, n, probability=True, epsilon=1e-8):
    original_weights = model.weights()
    bumped_weights = original_weights.copy()
    p0 = model.probability() if probability else model.value()
    bumped_weights[n] = bumped_weights[n] + epsilon
    model.set_weights(bumped_weights)
    p1 = model.probability() if probability else model.value()
    model.set_weights(original_weights)
    gradient = [(pp1 - pp0) / epsilon for pp0, pp1 in zip(p0, p1)]
    return gradient
