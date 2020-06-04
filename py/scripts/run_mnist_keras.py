# From Hands-on machine learning page 299

# import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

# data
mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = [str(i) for i in range(10)]

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

print(model.predict(X_test[:10]))
print(model.predict_classes(X_test[:10]))
print(y_test[:10])
pass
