import random

from j2learn.etc.tools import finite_differences
from j2learn.function.function import reLU
from j2learn.layer.cnn import CNN
from j2learn.layer.dense import Dense
from j2learn.layer.category import Category
from j2learn.layer.image import Image
from j2learn.model.model import Model

pixels = 81
image = Image(image_data=[random.randint(59, 59) for _ in range(pixels)])
cnn_a = CNN(reLU(), (3, 3), name='a')
cnn_b = CNN(reLU(), (3, 3), name='b')
cnn_c = CNN(reLU(), (3, 3), name='c')
dense = Dense(reLU(), (10, 1), name='d')
category = Category([i for i in range(10)])
model = Model(layers=[
    image,
    cnn_a,
    cnn_b,
    cnn_c,
    dense,
    category,
])

model.compile(build=True)

v = model.value()
# v = model.probability()
analytic_jacobian = model.jacobian()
# finite_differences(model, False)
print(v)

model.update_data_layer([random.randint(199, 199) for _ in range(pixels)])
v = model.value()
# analytic_jacobian = model.jacobian()
finite_differences(model, False, nmax=500)
print(v)
pass
