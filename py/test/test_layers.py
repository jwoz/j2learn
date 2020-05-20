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
### test a non-convoluted CNN. Need the same output
identity_cnn = CNN(reLU(), (1, 1), (0, 0), image_layer, weight=1)
for i in range(identity_cnn.shape()[0] * identity_cnn.shape()[1]):
    assert identity_cnn.node(i).value() == identity_cnn._underlying_layer.node(i).value()
print(identity_cnn.display())

### test convolution
first_cnn = CNN(reLU(), (3, 3), (0, 0), identity_cnn)
print(first_cnn.display(threshold=0.5))
second_cnn = CNN(reLU(), (3, 3), (0, 0), first_cnn)
print(second_cnn.display(threshold=0.5))

### more layers
dense = Dense(reLU(), (10, 1), second_cnn)
category = Category([i for i in range(10)], dense)
print(category.node(0).value())
print(category.node(0).probability())

#### replace image
r = 388
print(f'>>>> {r} <<<<')
image_layer.set_image_data_and_label(image_data=images[r], label=labels[r])
print(image_layer.display(numbers=True))
# crucial steps:
identity_cnn.build(init=1)
first_cnn.build(init=1)
second_cnn.build(init=1)
print(identity_cnn.display(numbers=True))
print(first_cnn.display(numbers=True))
print(second_cnn.display(numbers=True))
