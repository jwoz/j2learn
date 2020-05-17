from abc import ABC, abstractmethod

class LayerBase(ABC):
    @abstractmethod
    def shape(self):
        ...

    @abstractmethod
    def node(self, i, j=None):
        ...


