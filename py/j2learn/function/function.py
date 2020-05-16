from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def value(self, z):
        ...

    @abstractmethod
    def derivative(self, z):
        ...


class reLU(ActivationFunction):
    def value(self, z):
        return max(0, z)

    def derivative(self, z):
        if z < 0: return 0
        if z > 0: return 1
        return np.nan()
