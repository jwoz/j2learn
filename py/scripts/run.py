import random

from j2learn.etc.tools import finite_differences
from j2learn.function.function import reLU
from j2learn.layer.cnn import CNN
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image
from j2learn.model.model import Model

image = Image(image_data=[random.randint(0, 255) for _ in range(25)])
image = Image(image_data=[0.1 for _ in range(25)])
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

model.compile(build=True)

v = model.value()
analytic_jacobian = model.jacobian()
finite_differences(model, False)
print(v)

new_image = Image(image_data=[random.randint(0, 255) for _ in range(25)])
new_image = Image(image_data=[0.2 for _ in range(25)])
model.update_data_layer([0.2 for _ in range(25)])
# model.build()
v = model.value()
# analytic_jacobian = model.jacobian()
# finite_differences(model, False)
print(v)
pass
