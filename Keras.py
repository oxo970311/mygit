import tensorflow as tf
from Scripts.pywin32_postinstall import verbose
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# print(train_input.shape,train_target.shape)
#
# fig, axs = plt.subplots(1,10,figsize=(10,10))
# for i in range(10):
#     axs[i].imshow(train_input[i],cmap='gray_r')
#     axs[i].axis('off')
#
# plt.show()

# train_scaled = train_input / 255.0
# train_scaled = train_scaled.reshape(-1, 28 * 28)
# train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2,
#                                                                       random_state=42)
# print(train_scaled.shape, train_target.shape)
# print(val_scaled.shape, val_target.shape)

# dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
# model = keras.Sequential([dense])
#
# model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# print(train_target[:10])
#
# model.fit(train_scaled, train_target, epochs=5)
# model.evaluate(val_scaled, val_target)


# dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden')
# dense2 = keras.layers.Dense(10, activation='softmax', name='output')
#
# model = keras.Sequential([dense1, dense2], name='패션 MNIST 모델')
# model.summary()
#
# model = keras.Sequential()
# model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
# model.add(keras.layers.Dense(10, activation='softmax'))
# model.summary()

train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2,
                                                                      random_state=42)
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_scaled, train_target, epochs=5, verbose=0)
model.evaluate(val_scaled, val_target)
model.summary()


def model_fn(a_layer=None):
    model = keras.Sequential(name='def_model')
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model


model = model_fn()
model.summary()

model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)

print(history.history.keys())

plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
