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
        self._image_data = [0] * shape[1] * shape[0] if image_data is None else [i / maximum for i in image_data]

    def set_image_data_and_label(self, image_data, label=None):
        self._image_data = [i / self._maximum for i in image_data]
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
            assert i < len(self._image_data)
            return DataNode(self._image_data[i], f'Data_{i}')
        assert i * self._shape[1] + j < len(
            self._image_data), f'{i}, {j}: {i * self._shape[1] + j}>{len(self._image_data)}'
        return DataNode(self._image_data[i * self._shape[1] + j], f'Data_{i * self._shape[1] + j}')

    def shape(self):
        return self._shape

    def display(self, numbers=False, threshold=0.8):
        render = ''
        for i in range(len(self._image_data)):
            if i % self._shape[0] == 0:
                render += '\n'
            if numbers:
                render += f'{self._image_data[i]:4.2f} ' if self._image_data[i] > 0.01 else '.... '
            elif self._image_data[i] > threshold:
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
