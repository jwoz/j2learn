from j2learn.data.mnist_images import MNISTData
from j2learn.function.function import reLU
from j2learn.layer.category import Category
from j2learn.layer.cnn import CNN
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image
from j2learn.model.model import Model
from j2learn.etc.tools import finite_difference

mndata = MNISTData(path='C:/Users/juergen/Dropbox/data/mnist')

images, labels = mndata.training()

r = 49  # a nice number three
image_layer = Image(image_data=images[r], label=labels[r])
cnn = CNN(reLU(), (3, 3), (0, 0))
dense = Dense(reLU(), (10, 1))
category = Category([i for i in range(10)])

model = Model(layers=[image_layer, cnn, dense, category])
model.compile(build=True)

cat = model.predict()
print(cat)
prob = model.probability()
print(prob)

n_weights = model.weight_count()

gradient = 0
i = 0
while gradient == 0 and i < n_weights:
    gradient = finite_difference(model, i, epsilon=0.01)
    i += 1
print(gradient)

weights = model.weights()
print(weights)

weight_counts = model.weight_counts()
print(weight_counts)

n_weights = model.weight_count()

new_weights = [-i - 1 for i in range(n_weights)]
model.set_weights(new_weights)

weights = model.weights()
print(weights)

pass
