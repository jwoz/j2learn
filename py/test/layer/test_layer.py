from unittest import TestCase

from j2learn.data.mnist_images import MNISTData
from j2learn.etc.tools import finite_differences, flatten
from j2learn.function.function import reLU
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image
from j2learn.model.model import Model


class TestLayerBase(TestCase):
    def test_jacobian(self):
        mndata = MNISTData(path='../mnist')
        images, labels = mndata.training()

        r = 49  # a nice number three
        image_layer = Image(image_data=images[r], label=labels[r])
        dense = Dense(reLU(), (1, 1))

        model = Model(layers=[image_layer, dense])
        model.compile(build=True)
        weight_count = model.weight_count()

        # bump and grind
        gradients = finite_differences(model, False)
        jacobian = list(flatten(dense.jacobian()))
        mod_jacb = list(flatten(model.jacobian()))
        assert len(gradients) == len(jacobian)
        for g, j, m in zip(gradients.values(), jacobian, mod_jacb):
            print(f'{g[0]:8.6f} {j:8.6f} {m:8.6f}')
        for g, j, m in zip(gradients.values(), jacobian, mod_jacb):
            self.assertAlmostEqual(g[0], j, delta=0.001)
            self.assertAlmostEqual(j, m, delta=0.000001)
