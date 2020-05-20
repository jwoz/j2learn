from j2learn.data.mnist_images import MNISTData
from j2learn.function.function import reLU
from j2learn.layer.category import Category
from j2learn.layer.cnn import CNN
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image
from j2learn.model.model import Model

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

weights = model.weights()
print(weights)

weight_counts = model.weight_counts()
print(weight_counts)
pass