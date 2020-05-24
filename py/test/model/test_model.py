import random
from unittest import TestCase

from j2learn.etc.tools import flatten, finite_differences
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
        # are these weights a weakref???
        analytic_jacobian = model.jacobian()
        flattened_analytic_jacobian = list(flatten(analytic_jacobian))
        if expected_no_partials is not None:
            self.assertEqual(len(flattened_analytic_jacobian), expected_no_partials)

        # go through the weights above and bump each. not this arbitrary list.
        bumped_derivatives = finite_differences(model, False)
        for w in model.weights(flatten=True, reset=False):
            a = w.derivative()
            b = bumped_derivatives[w.id]
            self.assertAlmostEqual(a, b[0], 4)
            self.assertEqual(len(b), 1)

            # for aa, bb in zip(a, b):
            #     self.assertAlmostEqual(aa, bb, 4)
        print(flattened_analytic_jacobian)

    def test_jacobian_tiny(self):  # passes
        image = [random.randint(0, 255) for _ in range(1)]
        small_image = Image(image_data=image)
        dense = Dense(reLU(), (1, 1))
        cnn = CNN(reLU(), (1, 1), (0, 0))
        model = Model(layers=[small_image, cnn, dense])
        self._rum_derivatives_test(
            model
        )

    def test_jacobian_tiny_dense(self):  # passes
        image = [random.randint(0, 255) for _ in range(2)]
        image = Image(image_data=image, shape=(2, 1))
        dense = Dense(reLU(), (1, 1))
        model = Model(layers=[image, dense])
        self._rum_derivatives_test(
            model
        )

    def test_jacobian_tiny_cnn(self):  # passes
        image = [random.randint(0, 255) for _ in range(2)]
        image = Image(image_data=image, shape=(2, 1))
        cnn = CNN(reLU(), (1, 1), (0, 0))
        model = Model(layers=[image, cnn])
        self._rum_derivatives_test(
            model
        )

    def test_jacobian_small(self):  # passes
        image = Image(image_data=[random.randint(0, 255) for _ in range(4)])
        dense = Dense(reLU(), (2, 1))
        cnn = CNN(reLU(), (1, 2), (0, 0))
        model = Model(layers=[
            image,
            cnn,
            dense])
        self._rum_derivatives_test(
            model
        )

    def test_jacobian_cnn(self):  # passes
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

    def test_jacobian_two_dense(self):  # passes
        image = Image(image_data=[random.randint(0, 255) for _ in range(2)], shape=(1, 2))
        dense_a = Dense(reLU(), (1, 2))
        dense_b = Dense(reLU(), (1, 2))
        model = Model(layers=[
            image,
            dense_a,
            dense_b,
        ])
        self._rum_derivatives_test(
            model
        )

    def test_jacobian_three_dense(self):  # fails
        random.seed(42)
        image = Image(image_data=[random.randint(0, 255) for _ in range(2)], shape=(1, 2))
        dense_a = Dense(reLU(), (1, 2), name='a')
        dense_b = Dense(reLU(), (1, 2), name='b')
        dense_c = Dense(reLU(), (1, 1), name='c')
        model = Model(layers=[
            image,
            dense_a,
            dense_b,
            dense_c,
        ])
        self._rum_derivatives_test(
            model
        )

    def test_jacobian_three_cnn(self):  # fails
        image = [random.randint(0, 255) for _ in range(2)]
        image = Image(image_data=image, shape=(1, 2))
        cnn_a = CNN(reLU(), (1, 2), (0, 0))
        cnn_b = CNN(reLU(), (1, 2), (0, 0))
        cnn_c = CNN(reLU(), (1, 1), (0, 0))
        model = Model(layers=[
            image,
            cnn_a,
            cnn_b,
            cnn_c,
        ])
        self._rum_derivatives_test(
            model
        )
