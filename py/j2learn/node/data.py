from dataclasses import dataclass


@dataclass
class DataNode:
    _value: float
    _name: str = ''

    def __str__(self):
        return self._name

    @staticmethod
    def weight_count():
        return 0.0

    def value(self):
        return self._value

    @staticmethod
    def derivative():
        return 0.0


class ZeroNode(DataNode):
    def __init__(self):
        super().__init__(0.0)
