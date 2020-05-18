import math

from j2learn.node.data import DataNode


class Image:
    def __init__(self, image_data=None, shape=None, label=None):
        assert image_data is not None or shape is not None
        self._label = label

        if shape is None:
            shape = (int(math.sqrt(len(image_data))),) * 2
        assert image_data is None or shape[0] * shape[1] == len(image_data)
        self._shape = shape
        self._image_data = [0] * shape[1] * shape[0] if image_data is None else image_data

    def set_image_data_and_label(self, image_data, label=None):
        self._image_data = image_data
        self._label = label

    def label(self):
        return self._label

    def node(self, i, j=None):
        if j is None:
            assert i < len(self._image_data)
            return DataNode(self._image_data[i])
        assert i + self._shape[0] * j < len(self._image_data), f'{i}, {j}: {i + self._shape[0] * j}>{len(self._image_data)}'
        return DataNode(self._image_data[i + self._shape[0] * j])

    def shape(self):
        return self._shape

    def display(self, threshold=200):
        render = ''
        for i in range(len(self._image_data)):
            if i % self._shape[0] == 0:
                render += '\n'
            if self._image_data[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render

    def jacobian(self):
        return [[]]
