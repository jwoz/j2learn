from j2learn.data.mnist_images import MNISTData
from j2learn.function.function import reLU
from j2learn.layer.category import Category
from j2learn.layer.cnn import CNN
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image

mndata = MNISTData(path='C:/Users/juergen/Dropbox/data/mnist')

images, labels = mndata.training()

r = 49  # a nice number three
image_layer = Image(image_data=images[r], label=labels[r])
print(image_layer.display())
first_cnn = CNN(reLU(), (1, 1), (0, 0), image_layer, weight=1)
print(first_cnn.display())


####################
image_layer = Image(shape=(28, 28))
first_cnn = CNN(reLU(), (3, 3), (0, 0), image_layer)
second_cnn = CNN(reLU(), (3, 3), (0, 0), first_cnn)
dense = Dense(reLU(), (10, 1), second_cnn)
category = Category([i for i in range(10)], dense)

print(category.node(1).value())

# r = random.randint(0, 500)
r = 49  # a nice number three
image_layer.set_image_data_and_label(images[r], labels[r])
print(image_layer.display())

first_cnn.build()
print(first_cnn.node(155).value())

print(first_cnn.node(16, 5)._weights)
print(first_cnn.node(16, 5)._underlying_nodes)
print(category.node(1).value())
