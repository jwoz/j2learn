from dataclasses import dataclass


@dataclass
class DataNode:
    _value: float
    _derivative: float = 0

    @staticmethod
    def weight_count():
        return 0

    def value(self):
        return self._value

    def derivative(self):
        return self._derivative


class ZeroNode(DataNode):
    def __init__(self):
        super().__init__(0)
