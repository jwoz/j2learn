from abc import ABC, abstractmethod

class LayerBase(ABC):
    @abstractmethod
    def shape(self):
        ...

    @abstractmethod
    def value(self, x, y):
        ...

    @abstractmethod
    def node(self, i):
        ...


