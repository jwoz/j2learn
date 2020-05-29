from j2learn.data.mnist_images import MNISTData
from j2learn.function.function import reLU
from j2learn.layer.category import Category
from j2learn.layer.cnn import CNN
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image
from j2learn.etc.tools import reduce as reduce_image

mndata = MNISTData(path='..\\test\\mnist')

images, labels = mndata.training()

r = 49  # a nice number three
image_layer = Image(image_data=images[r], label=labels[r])
print(image_layer.display(numbers=True))

r = 49  # a nice number three
reduced_image = reduce_image(images[r])
image_layer = Image(image_data=reduced_image, label=labels[r])
print(image_layer.display(numbers=True))
