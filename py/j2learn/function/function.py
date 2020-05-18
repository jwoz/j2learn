import numpy as np


class reLU:
    @staticmethod
    def value(z):
        return max(0, z)

    @staticmethod
    def derivative(z):
        if z < 0:
            return 0
        if z > 0:
            return 1
        return np.nan()
