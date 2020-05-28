import random
from j2learn.function.function import reLU
from j2learn.layer.cnn import CNN
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image
from j2learn.model.model import Model
from j2learn.etc.tools import flatten, finite_differences

image = [random.randint(0, 255) for _ in range(25)]
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


model.compile(build=True)

analytic_jacobian = model.jacobian()
model.value()


finite_differences(model, False)
