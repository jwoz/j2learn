import numpy as np

def matrix_product(a, b, skip_nan=False):
    assert len(a[0]) == len(b)
    m = []
    for i in range(len(a)):
        r = []
        for k in range(len(b[0])):
            this_sum = 0
            for j in range(len(a[0])):
                s = a[i][j] * b[j][k]
                if skip_nan and np.isnan(s):
                    continue
                this_sum += s
            r.append(this_sum)
        m.append(r)
    assert len(m) == len(a) and len(m[0]) == len(b[0])
    return m
