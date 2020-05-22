import random
from unittest import TestCase

from j2learn.etc.tools import flatten, finite_difference
from j2learn.function.function import reLU
from j2learn.layer.cnn import CNN
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image
from j2learn.model.model import Model


class TestModel(TestCase):
    def _rum_derivatives_test(self, model, expected_no_weights=None, expected_no_partials=None):
        model.compile(build=True)
        weight_count = model.weight_count()
        if expected_no_weights is not None:
            self.assertEqual(weight_count, expected_no_weights)

        mod_jacb = list(flatten(model.jacobian()))
        if expected_no_partials is not None:
            self.assertEqual(len(mod_jacb), expected_no_partials)

        bumped_derivatives = []
        for i in range(weight_count):
            d = finite_difference(model, i, False)
            bumped_derivatives.append(d)
        non_zero_bumped_derivatives = [b for b in flatten(bumped_derivatives) if b != 0]
        for m, b in zip(mod_jacb, non_zero_bumped_derivatives):
            self.assertAlmostEqual(m, b, 4)
        print(mod_jacb)

    def test_jacobian_small(self):
        image = [random.randint(0, 255) for _ in range(4)]
        small_image = Image(image_data=image)
        dense = Dense(reLU(), (2, 1))
        cnn = CNN(reLU(), (1, 2), (0, 0))
        model = Model(layers=[small_image, cnn, dense])
        self._rum_derivatives_test(
            model,
            14,
            20  # + 8 zerp derivatives
        )

    def test_jacobian_cnn(self):
        image = [random.randint(0, 255) for _ in range(2)]
        image = Image(image_data=image, shape=(1, 2))
        cnn_a = CNN(reLU(), (1, 1), (0, 0))
        cnn_b = CNN(reLU(), (1, 1), (0, 0))
        model = Model(layers=[
            image,
            cnn_a,
            cnn_b,
        ])
        self._rum_derivatives_test(
            model
        )
