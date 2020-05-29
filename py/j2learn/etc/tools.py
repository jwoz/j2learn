import math
from collections import Iterable
from timeit import default_timer

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


def finite_differences(model, probability=True, nonzero=True, epsilon=1e-8, nmax=0):
    weights = model.weights()
    p0 = model.probability() if probability else model.value()
    gradients = {}
    n = len(weights)
    j = 0
    t0 = default_timer()
    for i, w in enumerate(weights):
        if 0 < nmax < i:
            break
        if i % 100 == 0:
            if i > 0:
                dt = default_timer() - t0
                j += i + 1
                tt = dt / i * n
                print(f' {dt / 60:6.2f}/{tt / 60:6.2f} min')
            print(f'{i}/{n} ', end='', flush=True)
        elif i % 10 == 0:
            print('+', end='', flush=True)

        if isinstance(w, ZeroWeight):
            continue
        original = w.weight()
        model.set_weight(w, original + epsilon)
        p1 = model.probability() if probability else model.value()
        gradient = [(pp1 - pp0) / epsilon for pp0, pp1 in zip(p0, p1)]
        if nonzero:
            gradient = [g for g in gradient]
        gradients[w.id] = gradient
        model.set_weight(w, original)
    print('')
    return gradients


def reduce(image):
    n = int(math.sqrt(len(image)))
    assert n * n == len(image)
    r = []
    for i in range(0, n - 1, 2):
        for j in range(0, n - 1, 2):
            v1 = image[i * n + j]
            v2 = image[i * n + j + 1]
            v3 = image[(i + 1) * n + j]
            v4 = image[(i + 1) * n + j + 1]
            r.append((v1 + v2 + v3 + v4) / 4.0)
    return r
