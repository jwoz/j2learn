from collections import Iterable


def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

# also flatten in pandas ... import seems slow
# from pandas.core.common import flatten