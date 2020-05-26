from collections import Iterable

from j2learn.node.weight import ZeroWeight


def flatten(items):
    # also flatten in pandas ... import seems slow
    # from pandas.core.common import flatten
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def finite_differences(model, probability=True, nonzero=True, epsilon=1e-8):
    weights = model.weights()
    p0 = model.probability() if probability else model.value()
    gradients = {}
    for w in weights:
        if isinstance(w, ZeroWeight):
            continue
        original = w.weight()
        w.set_weight(original + epsilon)
        p1 = model.probability() if probability else model.value()
        gradient = [(pp1 - pp0) / epsilon for pp0, pp1 in zip(p0, p1)]
        if nonzero:
            gradient = [g for g in gradient]
        gradients[w.id] = gradient
        w.set_weight(original)
    return gradients
