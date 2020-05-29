from mnist import MNIST

from j2learn.etc.tools import reduce as reduce_image


class MNISTData:
    def __init__(self, path, gz=True, reduce=False):
        self._mndata = MNIST(path, gz=gz)
        self._reduce = reduce

    def training(self):
        images, labels = self._mndata.load_training()
        if not self._reduce:
            return images, labels
        images = [reduce_image(i) for i in images]
        return images, labels

    def testing(self):
        images, labels = self._mndata.load_testing()
        return images, labels
