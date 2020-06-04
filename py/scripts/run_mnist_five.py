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

mndata = MNISTData(path='../test/mnist')
predict_labels =[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

activation = reLU()
model = Model(layers=[
    Image(shape=(14, 14)),
    Dense(activation, (10, 1), name='d1'),
    # Dense(activation, (10, 1), name='d2'),
    SoftMax([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name='s1'),
])
model.compile(build=True)

# Gradient Descent
gd = GradientDescent(model=model, learning_rate=1.0, labels=predict_labels)

# train
images, labels = mndata.training()
reduced_images = []
reduced_labels = []
i = 0
while i < len(images) and len(reduced_images) < 5000:
    if labels[i] not in predict_labels:
        i += 1
        continue
    reduced_images.append(reduce_image(images[i]))
    reduced_labels.append(labels[i])
    i += 1

iterations = 10000
gd.sgd(reduced_images, reduced_labels, iterations=iterations)

# test the model
test_images, test_labels = mndata.testing()

predictions = []
for i, (ti, tl) in enumerate(zip(test_images, test_labels)):
    if tl not in predict_labels:
        continue
    model.update_data_layer(reduce_image(ti))
    p = model.predict()
    if i % 100 == 0:
        print(f'{p}, {tl}, {max(model.value()):6.4f}')
    predictions.append(dict(label=tl, predicted_label=p, probability=max(model.value())))
predictions = pd.DataFrame(predictions)
predictions.to_csv(f'five_{datetime.datetime.now():%Y%m%dT%H%M}.csv')
