from j2learn.data.mnist_images import MNISTData
from j2learn.etc.tools import finite_difference, flatten
from j2learn.function.function import reLU
from j2learn.layer.dense import Dense
from j2learn.layer.image import Image
from j2learn.model.model import Model

mndata = MNISTData(path='C:/Users/juergen/Dropbox/data/mnist')

images, labels = mndata.training()

r = 49  # a nice number three
image_layer = Image(image_data=images[r], label=labels[r])
dense = Dense(reLU(), (1, 1))

model = Model(layers=[image_layer, dense])
model.compile(build=True)
weight_count = model.weight_count()

# bump and grind
gradients = []
i = 0
while i < weight_count:
    gradient = finite_difference(model, i, probability=False, epsilon=0.01)
    gradients.append(gradient)
    i += 1
jacobian = list(flatten(dense.jacobian()))
assert len(gradients) == len(jacobian)
for g, j in zip(gradients, jacobian):
    assert abs(g - j) < 0.001

nonzero_jacobian = [(i, j) for i, j in enumerate(jacobian) if j != 0]
print(nonzero_jacobian)

cat = model.predict()
print(cat)
prob = model.probability()
print(prob)

weights = model.weights()
print(weights)

weight_counts = model.weight_counts()
print(weight_counts)

weight_count = model.weight_count()

new_weights = [-i - 1 for i in range(weight_count)]
model.set_weights(new_weights)

weights = model.weights()
print(weights)

pass
