import math

from j2learn.node.data import DataNode


class Image:
    is_root = True

    def __init__(self, image_data=None, shape=None, label=None, maximum=256):
        assert image_data is not None or shape is not None
        self._label = label

        if shape is None:
            shape = (int(math.sqrt(len(image_data))),) * 2
        assert image_data is None or shape[0] * shape[1] == len(image_data)
        self._shape = shape
        self._maximum = maximum
        image_data = [0] * shape[1] * shape[0] if image_data is None else [i / maximum for i in image_data]
        self._nodes = [DataNode(i,  f'Data_{j}') for j, i in enumerate(image_data)]

    def set_image_data_and_label(self, image_data, label=None):
        for n, i in zip(self._nodes, image_data):
            n.set_value(i)
        self._label = label

    @staticmethod
    def set_weights(weights):
        assert not len(weights)

    @staticmethod
    def build():
        return

    def label(self):
        return self._label

    def node(self, i, j=None):
        if j is None:
            assert i < len(self._nodes)
            return self._nodes[i]
        assert i * self._shape[1] + j < len(self._nodes), f'{i}, {j}: {i * self._shape[1] + j}>{len(self._nodes)}'
        return self._nodes[i * self._shape[1] + j]

    def shape(self):
        return self._shape

    def display(self, numbers=False, threshold=0.8):
        render = ''
        for i in range(len(self._nodes)):
            if i % self._shape[0] == 0:
                render += '\n'
            if numbers:
                render += f'{self._nodes[i].value():4.2f} ' if self._nodes[i].value() > 0.01 else '.... '
            elif self._nodes[i].value() > threshold:
                render += '@'
            else:
                render += '.'
        return render

    @staticmethod
    def weight_count():
        return 0

    @staticmethod
    def weights():
        return []

    def value(self, cache=None):
        return self._image_data

    @staticmethod
    def chain_rule_factors(upper_layer_factors=None, cache=None):
        return upper_layer_factors

    @staticmethod
    def jacobian(factors=None, cache=None):
        return []
