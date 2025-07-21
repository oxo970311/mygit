import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()

(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist
x_train, y_train = x_train_full[:-5000], y_train_full[:-5000]
x_valid, y_valid = x_train_full[-5000:], y_train_full[-5000:]


x_train, x_valid, x_test = x_train / 255., x_valid / 255., x_test / 255.
class_name = ["T_shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
print(class_name[y_train[0]])

tf.random.set_seed(42)
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=[28, 28]))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(300, activation="relu"))
# model.add(tf.keras.layers.Dense(100, activation="relu"))
# model.add(tf.keras.layers.Dense(10, activation="softmax"))
# model.summary()
#
# hidden1 = model.layers[1]
# print(hidden1.name)
#
# weights, biases = hidden1.get_weights()
#
# print(weights, weights.shape)
# print(biases, biases.shape)
#
# model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
#
# history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))
# pd.DataFrame(history.history).plot(figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="epochs",
#                                    style=["r--", "r-*", "b-", "b-*"])
# plt.show()
# print(model.evaluate(x_test, y_test))

# norm_layer = tf.keras.layers.Normalization(input_shape=x_train.shape[1:])
# norm_layer.adapt(x_train)
# model = tf.keras.Sequential([
#     norm_layer,
#     tf.keras.layers.Dense(50, activation="relu"),
#     tf.keras.layers.Dense(50, activation="relu"),
#     tf.keras.layers.Dense(50, activation="relu"),
#     tf.keras.layers.Dense(1)
# ])
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
# history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))
# mse_test, rmse_test = model.evaluate(x_test, y_test)
# x_new = x_test[:3]
# y_pred = model.predict(x_new)

# normalization_layer = tf.keras.layers.Normalization()
# hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
# hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
# concat_layer = tf.keras.layers.Concatenate()
# output_layer = tf.keras.layers.Dense(1)
#
# input_ = tf.keras.layers.Input(shape=x_train.shape[1:])
# normalized = normalization_layer(input_)
# hidden1 = hidden_layer1(normalized)
# hidden2 = hidden_layer2(hidden1)
# concat = concat_layer([normalized, hidden2])
# output = output_layer(concat)
#
# model.fit = tf.keras.model(inputs=[input_], outputs=[output])

# #page=405
# input_wide = tf.keras.layers.Input(shape=[784])
# input_deep = tf.keras.layers.Input(shape=[784])
# norm_layer_wide = tf.keras.layers.Normalization()
# norm_layer_deep = tf.keras.layers.Normalization()
# norm_wide = norm_layer_wide(input_wide)
# norm_deep = norm_layer_deep(input_deep)
# hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
# hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
# concat = tf.keras.layers.concatenate([norm_wide, hidden2])
# output = tf.keras.layers.Dense(1)(concat)
# model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])
# print(norm_wide.shape, norm_deep.shape)
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
# x_train = x_train.reshape([-1, 28 * 28])
# x_valid = x_valid.reshape([-1, 28 * 28])
# x_test = x_test.reshape([-1, 28 * 28])
# target = y_test[:10000]
# x_train_wide, x_train_deep = x_train[:, :784], x_train[:, :784]
# x_valid_wide, x_valid_deep = x_valid[:, :784], x_valid[:, :784]
# x_test_wide, x_test_deep = x_test[:, :784], x_test[:, :784]
# x_new_wide, x_new_deep = x_test_wide[:3], x_test_deep[:3]
# print(x_train_wide.shape, x_train_deep.shape, x_test_wide.shape, x_new_wide.shape)
# norm_layer_wide.adapt(x_train_wide)
# norm_layer_deep.adapt(x_train_deep)
# history = model.fit((x_train_wide, x_train_deep), y_train, epochs=20,
#                     validation_data=((x_valid_wide, x_valid_deep), y_valid))
# mse_test = model.evaluate((x_test_wide, x_test_deep), target)
# y_pred = model.predict((x_train_wide, x_train_deep))
# print(y_pred)

class WideAndDeepModel(tf.keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)  # 모델 이름을 지정하는 데 필요합니다
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)

    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output

# tf.random.set_seed(42)  # 추가 코드 - 재현성을 위한 것
# model = WideAndDeepModel(30, activation="relu", name="my_cool_model")
#
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# model.compile(loss="mse", loss_weights=[0.9, 0.1], optimizer=optimizer,
#               metrics=["RootMeanSquaredError", "RootMeanSquaredError"])
# model.norm_layer_wide.adapt(x_train_wide)
# model.norm_layer_deep.adapt(x_train_deep)
# history = model.fit(
#     (x_train_wide, x_train_deep), (y_train, y_train), epochs=10,
#     validation_data=((x_valid_wide, x_valid_deep), (y_valid, y_valid)))
# eval_results = model.evaluate((x_test_wide, x_test_deep), (y_test, y_test))
# weighted_sum_of_losses, main_rmse, main_rmse, aux_rmse = eval_results
# y_pred_main, y_pred_aux = model.predict((x_new_wide, x_new_deep))