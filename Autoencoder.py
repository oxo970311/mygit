import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

(train_input, train_target), (test_input, test_target) = tf.keras.datasets.fashion_mnist.load_data()

train_input = train_input.astype("float32") / 255.0

print(train_input.shape, train_target.shape)

# fig, axs = plt.subplots(1, 10, figsize=(10, 10))
# for i in range(10):
#     axs[i].imshow(train_input[i], cmap='gray_r')
#     axs[i].axis('off')
#
# plt.show()

# fig, axs = plt.subplots(10, 10, figsize=(10, 10))
# for i in range(10):
#     for j in range(10):
#         axs[i][j].imshow(train_input[i], cmap='gray_r')
#         axs[i][j].axis('off')
#
# plt.show()

x_train, x_valid, y_train, y_valid = train_test_split(train_input[:5000], train_target[:5000], test_size=0.2,
                                                      random_state=42)

print(x_train.shape)
print(y_valid.shape)

st_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu")
])

st_decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(28 * 28),
    tf.keras.layers.Reshape([28, 28])
])

at_encoder = tf.keras.Sequential([st_encoder, st_decoder])
at_encoder.compile(loss="mse", optimizer="nadam")
history = at_encoder.fit(x_train, x_train, epochs=1, validation_data=(x_valid, x_valid))


def plot_reconstructions(model, images=x_valid, n_images=5):
    reconstructions = np.clip(model.predict(images[:n_images]), 0, 1)
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plt.imshow(images[image_index], cmap="binary")
        plt.axis("off")
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plt.imshow(reconstructions[image_index], cmap="binary")
        plt.axis("off")


plot_reconstructions(at_encoder)
plt.show()

x_valid_compressed = st_encoder.predict(x_valid)
tsne = TSNE(init="pca", learning_rate="auto", random_state=42)
x_valid_2d = tsne.fit_transform(x_valid_compressed)

print(x_valid_2d.shape)

plt.scatter(x_valid_2d[:, 0], x_valid_2d[:, 1], c=y_valid, s=10, cmap="tab10")
plt.show()

# class DenseTranspose(tf.keras.layers.Layer):
#     def __init__(self, dense, activation=None, **kwargs):
#         super().__init__(**kwargs)
#         self.dense = dense
#         self.activation = tf.keras.activations.get(activation)
#
#     def build(self, batch_input_shape):
#         self.biases = self.add_weight(name="bias", shape=self.dense.input_shape[-1], initializer="zeros")
#         super().build(batch_input_shape)
#
#     def call(self, inputs):
#         Z = tf.matmul(inputs, self.dense_weights[0], transpose_b=True)
#         return self.activation(Z + self.biases)
#
#
# dense_1 = tf.keras.layers.Dense(100, activation="relu")
# dense_2 = tf.keras.layers.Dense(30, activation="relu")
#
# tied_encoder = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     dense_1,
#     dense_2
# ])
#
# tied_decoder = tf.keras.Sequential([
#     DenseTranspose(dense_2, activation="relu"),
#     DenseTranspose(dense_1),
#     tf.keras.layers.Reshape([28, 28])
# ])
#
# tied_at = tf.keras.Sequential([tied_encoder, tied_decoder])
# tied_at.compile(loss="mse", optimizer="nadam")
# tied_at.fit(x_train, x_train, epochs=5, validation_data=(x_valid, x_valid))
# print(tied_at.evaluate(x_train, x_train))

conv_encoder = tf.keras.Sequential([
    tf.keras.layers.Reshape([28, 28, 1]),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(30, 3, padding="same", activation="relu"),
    tf.keras.layers.GlobalAvgPool2D()
])

conv_decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(3 * 3 * 16),
    tf.keras.layers.Reshape((3, 3, 16)),
    tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu"),
    tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu"),
    tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding="same"),
    tf.keras.layers.Reshape([28, 28])
])

conv_at = tf.keras.Sequential([conv_encoder, conv_decoder])
conv_at.compile(loss="mse", optimizer="nadam")
conv_at.fit(x_train, x_train, epochs=1, validation_data=(x_valid, x_valid))

# plot_reconstructions(conv_at)
# plt.show()

codings_size = 30

Dense = tf.keras.layers.Dense
generator = tf.keras.Sequential([
    Dense(100, activation="relu", kernel_initializer="he_normal"),
    Dense(150, activation="relu", kernel_initializer="he_normal"),
    Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    Dense(150, activation="relu", kernel_initializer="he_normal"),
    Dense(100, activation="relu", kernel_initializer="he_normal"),
    Dense(1, activation="sigmoid"),
])

gan = tf.keras.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

do_encoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu")
])

do_decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(28 * 28),
    tf.keras.layers.Reshape([28, 28])
])

at_dropout = tf.keras.Sequential([st_encoder, st_decoder])
at_dropout.compile(loss="mse", optimizer="nadam")
at_dropout.fit(x_train, x_train, epochs=1, validation_data=(x_valid, x_valid))

plot_reconstructions(at_dropout)
plt.show()


# class Sampling(tf.keras.layers.Layer):
#     def call(self, inputs):
#         mean, log_var = inputs
#         return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean
#
#
# codings_size = 10
#
# inputs = tf.keras.layers.Input(shape=[28, 28])
# Z = tf.keras.layers.Flatten()(inputs)
# Z = tf.keras.layers.Dense(150, activation="relu")(Z)
# Z = tf.keras.layers.Dense(100, activation="relu")(Z)
# codings_mean = tf.keras.layers.Dense(codings_size)(Z)
# codings_log_var = tf.keras.layers.Dense(codings_size)(Z)
# codings = Sampling()([codings_mean, codings_log_var])
# variational_encoder = tf.keras.Model(inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])
#
# decoder_inputs = tf.keras.layers.Input(shape=[codings_size])
# x = tf.keras.layers.Dense(100, activation="relu")(decoder_inputs)
# x = tf.keras.layers.Dense(150, activation="relu")(x)
# x = tf.keras.layers.Dense(28 * 28)(x)
# outputs = tf.keras.layers.Reshape([28, 28])(x)
# variational_decoder = tf.keras.Model(inputs=(decoder_inputs), outputs=[outputs])
#
# codings = variational_encoder(inputs)[2]
# recostructions = variational_decoder(codings)
# variational_at = tf.keras.Model(inputs=[inputs], outputs=[recostructions])
#
# # latent_loss = -0.5 * tf.reduce_sum(1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean), axis=-1)
# # variational_at.add_loss(tf.reduce_mean(latent_loss) / 784.) # error 원인 못찾음
#
# variational_at.compile(loss="mse", optimizer="nadam")
# history = variational_at.fit(x_train, x_train, epochs=25, batch_size=128, validation_data=(x_valid, x_valid))
#
# plot_reconstructions(variational_at)
# plt.show()