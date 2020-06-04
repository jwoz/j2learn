import datetime

import pandas as pd

from j2learn.data.mnist_images import MNISTData
from j2learn.etc.tools import reduce as reduce_image
from j2learn.function.function import reLU
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image
from j2learn.layer.softmax import SoftMax
from j2learn.model.model import Model
from j2learn.regression.gradient_descent import GradientDescent

# ## Use reduced image size?
reduce = True
if reduce:
    nx = ny = 14
else:
    nx = ny = 28

activation = reLU()

# ## Define model
predict_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
model = Model(layers=[
    Image(shape=(nx, ny)),
    Dense(activation, (100, 1), name='d1'),
    SoftMax(predict_labels, name='s1'),
])
model.compile(build=True)

# ## Prepare images for training
mndata = MNISTData(path='../test/mnist')
images, labels = mndata.training()
train_images = []
train_labels = []
i = 0
train_n = len(images)
while i < len(images) and len(train_images) < train_n:
    train_images.append(reduce_image(images[i]) if reduce else images[i])
    train_labels.append(labels[i])
    i += 1

# ## Stochastic Gradient Descent
gd = GradientDescent(model=model, learning_rate=0.1, labels=predict_labels)
gd.sgd(train_images, train_labels, iterations=20000)

# ## Test the model ###
test_images, test_labels = mndata.testing()

predictions = []
for i, (ti, tl) in enumerate(zip(test_images, test_labels)):
    model.update_data_layer(reduce_image(ti) if reduce else ti)
    p = model.predict()
    if i % 100 == 0:
        print(f'{p}, {tl}, {max(model.value()):6.4f}')
    predictions.append(dict(label=tl, predicted_label=p, probability=max(model.value())))
predictions = pd.DataFrame(predictions)
predictions.to_csv(f'mnist_{datetime.datetime.now():%Y%m%dT%H%M}.csv')
