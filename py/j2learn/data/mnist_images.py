from mnist import MNIST


class MNISTData:
    def __init__(self, path='/home/j/data/mnist', gz=True):
        self._mndata = MNIST(path, gz=gz)

    def training(self):
        images, labels = self._mndata.load_training()
        return images, labels

    def testing(self):
        images, labels = self._mndata.load_testing()
        return images, labels
