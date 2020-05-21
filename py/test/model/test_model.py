import random
from unittest import TestCase

from j2learn.etc.tools import flatten, finite_difference
from j2learn.function.function import reLU
from j2learn.layer.cnn import CNN
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image
from j2learn.model.model import Model


class TestModel(TestCase):
    def test_jacobian_small(self):
        image = [random.randint(0, 255) for _ in range(4)]
        small_image = Image(image_data=image)
        dense = Dense(reLU(), (2, 1))
        cnn = CNN(reLU(), (1, 2), (0, 0))
        model = Model(layers=[small_image, cnn, dense])
        model.compile(build=True)
        weight_count = model.weight_count()
        self.assertEqual(weight_count, 14)

        mod_jacb = list(flatten(model.jacobian()))
        self.assertEqual(len(mod_jacb), 20)

        bumped_derivatives = []
        for i in range(weight_count):
            d = finite_difference(model, i, False)
            bumped_derivatives.append(d)

        print(mod_jacb)
