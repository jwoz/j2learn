import random
from timeit import default_timer
from unittest import TestCase

import numpy as np

from j2learn.etc.tools import flatten, finite_differences
from j2learn.function.function import reLU
from j2learn.layer.category import Category
from j2learn.layer.cnn import CNN
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image
from j2learn.layer.softmax import SoftMax
from j2learn.model.model import Model


class TestModel(TestCase):
    def _run_derivatives_test(self, model, expected_no_weights=None, expected_no_partials=None):
        model.compile(build=True)
        weight_count = model.weight_count()
        if expected_no_weights is not None:
            self.assertEqual(weight_count, expected_no_weights)
        t0 = default_timer()
        analytic_jacobian = model.jacobian()
        print(f'Analytic jacobian: {default_timer() - t0}')
        flattened_analytic_jacobian = list(flatten(analytic_jacobian))
        if expected_no_partials is not None:
            self.assertEqual(len(flattened_analytic_jacobian), expected_no_partials)

        # go through the weights above and bump each. not this arbitrary list.
        t0 = default_timer()
        bumped_derivatives = finite_differences(model, False)
        print(f'Bumped jacobian:     {default_timer() - t0}')
        model_weights = model.weights()
        for m in model_weights:
            d = ', '.join([f'{dd:8.6f}' for dd in m.derivative()])
            b = ', '.join([f'{dd:8.6f}' for dd in bumped_derivatives[m.id]])
            print(f'{m}, {m.id}: [{d}] vs [{b}]')
        for w in model_weights:
            a = w.derivative()
            b = bumped_derivatives[w.id]
            # if the last layer has more than one node, the bumping will produce zeros that are not in the analytic derivatives. Eliminate:
            if len(a) < len(b):
                b = [bb for bb in b if bb != 0]
            for aa, bb in zip(a, b):
                self.assertAlmostEqual(aa, bb, 4)
        print(flattened_analytic_jacobian)
        return model_weights

    def test_jacobian_dense_11_cnn_11(self):
        image = [random.randint(0, 255) for _ in range(1)]
        small_image = Image(image_data=image)
        dense = Dense(reLU(), (1, 1))
        cnn = CNN(reLU(), (1, 1), (0, 0))
        model = Model(layers=[small_image, cnn, dense])
        self._run_derivatives_test(
            model
        )

    def test_jacobian_dense_11(self):
        image = [random.randint(0, 255) for _ in range(2)]
        image = Image(image_data=image, shape=(2, 1))
        dense = Dense(reLU(), (1, 1))
        model = Model(layers=[image, dense])
        self._run_derivatives_test(
            model
        )

    def test_jacobian_cnn_11(self):
        image = [random.randint(0, 255) for _ in range(2)]
        image = Image(image_data=image, shape=(2, 1))
        cnn = CNN(reLU(), (1, 1), (0, 0))
        model = Model(layers=[image, cnn])
        self._run_derivatives_test(
            model
        )

    def test_jacobian_dense_21_cnn_12(self):
        image = Image(image_data=[random.randint(0, 255) for _ in range(4)])
        dense = Dense(reLU(), (2, 1))
        cnn = CNN(reLU(), (1, 2), (0, 0))
        model = Model(layers=[
            image,
            cnn,
            dense])
        self._run_derivatives_test(
            model
        )

    def test_jacobian_cnn_11_11(self):
        image = [random.randint(0, 255) for _ in range(2)]
        image = Image(image_data=image, shape=(1, 2))
        cnn_a = CNN(reLU(), (1, 1), (0, 0), name='a')
        cnn_b = CNN(reLU(), (1, 1), (0, 0), name='b')
        model = Model(layers=[
            image,
            cnn_a,
            cnn_b,
        ])
        self._run_derivatives_test(
            model
        )

    def test_jacobian_dense_12_11(self):
        image = Image(image_data=[random.randint(0, 255) for _ in range(2)], shape=(1, 2))
        dense_a = Dense(reLU(), (1, 2))
        dense_b = Dense(reLU(), (1, 1))
        model = Model(layers=[
            image,
            dense_a,
            dense_b,
        ])
        self._run_derivatives_test(
            model
        )

    def test_jacobian_dense_12_12(self):
        image = Image(image_data=[random.randint(0, 255) for _ in range(2)], shape=(1, 2))
        dense_a = Dense(reLU(), (1, 2), name='a')
        dense_b = Dense(reLU(), (1, 2), name='b')
        model = Model(layers=[
            image,
            dense_a,
            dense_b,
        ])
        self._run_derivatives_test(
            model
        )
        pass

    def test_jacobian_dense_12_12_11(self):
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
        self._run_derivatives_test(
            model
        )

    def test_jacobian_dense_101_51_softmax_5(self):
        random.seed(42)
        image = Image(image_data=[random.randint(0, 255) for _ in range(2)], shape=(1, 2))
        dense_a = Dense(reLU(), (10, 1), name='a')
        dense_b = Dense(reLU(), (5, 1), name='b')
        softmax = SoftMax([1, 2, 3, 4, 5], name='c')
        model = Model(layers=[
            image,
            dense_a,
            dense_b,
            softmax,
        ])
        self._run_derivatives_test(
            model
        )

    def test_jacobian_cnn_12_12_11(self):
        image = [random.randint(0, 255) for _ in range(2)]
        image = Image(image_data=image, shape=(1, 2))
        cnn_a = CNN(reLU(), (1, 2), (0, 0), name='a')
        cnn_b = CNN(reLU(), (1, 2), (0, 0), name='b')
        cnn_c = CNN(reLU(), (1, 1), (0, 0), name='c')
        model = Model(layers=[
            image,
            cnn_a,
            cnn_b,
            cnn_c,
        ])
        self._run_derivatives_test(
            model
        )

    def test_jacobian_cnn_31(self):
        image = [random.randint(0, 255) for _ in range(3)]
        model = Model(layers=[
            Image(image_data=image, shape=(3, 1)),
            CNN(reLU(), (3, 1), (0, 0), name='a'),
        ])
        weights = self._run_derivatives_test(model)
        self.assertEqual(len(weights), 7)

    def test_jacobian_cnn_31_dense_11(self):
        random.seed(44009)
        image = [random.randint(0, 255) for _ in range(3)]
        model = Model(layers=[
            Image(image_data=image, shape=(3, 1)),
            CNN(reLU(), (3, 1), name='a'),
            Dense(reLU(), (1, 1), name='d'),
        ])
        self._run_derivatives_test(model)

    def test_jacobian_cnn_33_33_33_dense_31(self):
        image = [random.randint(0, 255) for _ in range(16)]
        image = Image(image_data=image)
        cnn_a = CNN(reLU(), (3, 3), (0, 0), name='a')
        cnn_b = CNN(reLU(), (3, 3), (0, 0), name='b')
        cnn_c = CNN(reLU(), (3, 3), (0, 0), name='c')
        dense = Dense(reLU(), (3, 1), name='d')
        model = Model(layers=[
            image,
            cnn_a,
            cnn_b,
            cnn_c,
            dense,
        ])
        self._run_derivatives_test(
            model
        )

    def test_jacobian_cnn_33_33_33_dense_11(self):
        image = [random.randint(0, 255) for _ in range(49)]
        image = Image(image_data=image)
        cnn_a = CNN(reLU(), (3, 3), name='a')
        cnn_b = CNN(reLU(), (3, 3), name='b')
        cnn_c = CNN(reLU(), (3, 3), name='c')
        dense = Dense(reLU(), (1, 1), name='d')
        model = Model(layers=[
            image,
            cnn_a,
            cnn_b,
            cnn_c,
            dense,
        ])
        self._run_derivatives_test(model)

    def test_softmax_tiny(self):
        image_data = [0.2, 0.1]
        categories = [1, 2, 3]
        model = Model(layers=[
            Image(image_data=image_data, shape=(2, 1), maximum=1),
            SoftMax(categories),
        ])
        model.compile(build=True)
        v = model.value()
        softmax_value = model._layers[1]._nodes[0].value({})
        self.assertEqual(len(categories), len(softmax_value))
        softmax_derivative = model._layers[1]._nodes[0].derivative({})
        bumped_derivatives = finite_differences(model, False)
        for i in range(6):
            for j in range(3):
                self.assertAlmostEqual(softmax_derivative[i][j], list(bumped_derivatives.values())[i][j])
        pass

    def test_softmax_small(self):
        image_data = [0.5]
        categories = [1, 2]
        model = Model(layers=[
            Image(image_data=image_data, shape=(1, 1), maximum=1),
            Dense(reLU(), (1, 1)),
            SoftMax(categories),
        ])
        self._run_derivatives_test(model)

    def test_softmax(self):
        image_data = [0.2, 0.5, 0.3]
        categories = [1, 2, 4]
        model = Model(layers=[
            Image(image_data=image_data, shape=(3, 1), maximum=1),
            Dense(reLU(), (3, 1)),
            SoftMax(categories),
        ])
        self._run_derivatives_test(model)

    def test_jacobian_category_3(self):
        image_data = [0.2, 0.5, 0.3]
        categories = [1, 2, 4]
        model = Model(layers=[
            Image(image_data=image_data, shape=(3, 1), maximum=1),
            Category(categories),
        ])
        model.compile(build=True)
        v = model.value()
        self.assertEqual(v, [max(image_data)])
        p = model.predict()
        self.assertEqual(p, [categories[int(np.argmax(np.array(image_data)))]])
        pass

    def test_jacobian_dense_31_category_3(self):
        image_data = [0.2, 0.5, 0.3]
        categories = [1, 2, 4]
        model = Model(layers=[
            Image(image_data=image_data, shape=(3, 1), maximum=1),
            Dense(reLU(), (3, 1)),
            Category(categories),
        ])
        self._run_derivatives_test(model)
        v = model.value()

        # compute value manually for this image_data:
        manual_dense_layer = []
        for i in range(3):
            weights = model._layers[1]._nodes[i]._weights
            manual_dense_layer.append(sum([w.weight() * d for w, d in zip(weights, image_data)]))
        max_value = max(manual_dense_layer)
        imax_value = int(np.argmax(np.array(manual_dense_layer)))
        self.assertEqual(v, [max_value])

        p = model.predict()
        self.assertEqual(p, [categories[imax_value]])
        pass

    def test_mnist_image(self):
        image_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 44.25, 122.75,
                      126.5, 70.5, 0.0, 0.0, 0.0, 0.0, 0.0, 6.5, 1.5, 0.0, 16.75, 203.75, 249.5, 252.0, 252.0, 246.5,
                      214.75, 37.5, 0.0, 0.0, 0.0, 25.25, 6.0, 0.0, 7.0, 123.5, 166.75, 162.25, 201.25, 252.5, 181.5,
                      9.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 167.5, 248.75, 240.5, 129.0, 27.5, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 212.75, 252.5, 208.75, 26.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 34.25, 71.5, 214.75, 202.0, 31.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.25,
                      169.5, 167.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 148.25, 190.5, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 12.75, 210.0, 159.5, 65.75, 167.5, 209.75, 252.5, 78.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 3.25, 168.0, 252.5, 251.25, 238.25, 163.75, 58.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.75,
                      106.5, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        model = Model(layers=[
            Image(image_data=image_data, maximum=256),
            CNN(reLU(), (3, 3)),
            Dense(reLU(), (3, 1)),
            Category([1, 2, 3]),
        ])
        model.compile(build=True)
        model.jacobian()
        weights = model.weights()
        derivatives = [w.derivative() for w in weights]
        pass
