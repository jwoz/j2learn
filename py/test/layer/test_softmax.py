from unittest import TestCase

import numpy as np

from j2learn.etc.tools import finite_differences, flatten
from j2learn.layer.image import Image
from j2learn.layer.softmax import SoftMax
from j2learn.model.model import Model


class TestSoftMaxNode(TestCase):
    def test_jacobian(self):
        image_layer = Image(image_data=[0.1, 0.3, 0.7], shape=(3, 1))
        softmax = SoftMax([1, 2])

        model = Model(layers=[image_layer, softmax])
        model.compile(build=True)

        # bump and grind
        gradients = finite_differences(model, False)
        gradients = list(flatten(gradients.values()))
        jacobian = list(flatten(softmax.jacobian(cache={})))
        mod_jacb = list(flatten(model.jacobian()))
        assert len(gradients) == len(jacobian)
        for g, j, m in zip(gradients, jacobian, mod_jacb):
            print(f'{g:8.6f} {j:8.6f} {m:8.6f}')
        for g, j, m in zip(gradients, jacobian, mod_jacb):
            self.assertAlmostEqual(g, j, delta=0.001)
            self.assertAlmostEqual(j, m, delta=0.000001)

    def test_chain_rule_factors(self):
        image_data = [0.1, 0.3, 0.7]
        image_layer = Image(image_data=image_data, shape=(3, 1), maximum=1)
        softmax = SoftMax([1, 2])

        model = Model(layers=[image_layer, softmax])
        model.compile(build=True)
        # bump and grind
        p0 = model.value()
        epsilon = 1e-6
        crf_bumped = []
        for i in range(len(image_data)):
            model.update_data_layer([((d + epsilon) if i == j else d) for j, d in enumerate(image_data)], maximum=1)
            p1 = model.value()
            crf_bumped.append([(pp1 - pp0) / epsilon for pp0, pp1 in zip(p0, p1)])
        crf_bumped = np.array(crf_bumped).transpose()

        crf_layer = softmax.chain_rule_factors(cache={})
        crf_model = model.chain_rule_factors()

        for g, j, m in zip(flatten(crf_bumped), flatten(crf_layer), flatten(crf_model)):
            print(f'{g:8.6f} {j:8.6f} {m:8.6f}')
        for g, j, m in zip(flatten(crf_bumped), flatten(crf_layer), flatten(crf_model)):
            self.assertAlmostEqual(g, j, delta=0.001)
            self.assertAlmostEqual(j, m, delta=0.000001)
