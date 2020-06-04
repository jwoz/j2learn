import math

import numpy as np


class identity:
    @staticmethod
    def value(z):
        return z

    @staticmethod
    def derivative(z):
        return 1


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
        # print('reLU: derivative is nan at z=0')
        return np.nan


class tanh:
    @staticmethod
    def value(z):
        return math.tanh(z)

    @staticmethod
    def derivative(z):
        tanh_z = math.tanh(z)
        return 1 - tanh_z * tanh_z
