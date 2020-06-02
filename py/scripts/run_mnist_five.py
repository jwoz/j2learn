from j2learn.data.mnist_images import MNISTData
from j2learn.etc.tools import reduce as reduce_image
from j2learn.function.function import tanh, identity
from j2learn.layer.category import Category
from j2learn.layer.cnn import CNN
from j2learn.layer.dense import Dense
from j2learn.layer.softmax import SoftMax
from j2learn.layer.image import Image
from j2learn.model.model import Model
from j2learn.regression.gradient_descent import GradientDescent

mndata = MNISTData(path='../test/mnist')

images, labels = mndata.training()

r = 49  # a nice number three
reduced_image = reduce_image(images[r])

activation = tanh()
model = Model(layers=[
    Image(image_data=reduced_image, label=labels[r]),
    CNN(activation, (3, 3), name='a'),
    # CNN(activation, (3, 3), name='b'),
    # CNN(activation, (3, 3), name='c'),
    Dense(activation, (2, 1), name='d'),
    SoftMax([5], name='e'),
    # Category([i for i in range(10)]),
])

model.compile(build=True)

# ## Some values
v = model.value()
p = model.predict()
analytic_jacobian = model.jacobian()
print(v, p)

r = 25
reduced_image = reduce_image(images[r])
model.update_data_layer(reduced_image, label=labels[r])
v = model.value()
p = model.predict()
print(v, p)

# ## Gradient Descent
gd = GradientDescent(model=model, learning_rate=0.00001)

iterations = 20000
reduced_images = [reduce_image(images[i]) for i in range(2000)]

gd.sgd(reduced_images, labels[:2000], iterations=iterations)

for i in range(2001, 2100):
    model.update_data_layer(reduce_image(images[i]), label=labels[i])
    p = model.predict()
    print(f'{p[0]}, {labels[i]}, {model.value()[0]:6.4f}')
