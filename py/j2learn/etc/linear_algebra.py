import numpy as np


def matrix_product(a, b):
    ma = np.array(a)
    mb = np.array(b)
    return np.matmul(ma, mb)
