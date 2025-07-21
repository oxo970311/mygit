import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

shakespeare_url = "https://homl.info/shakespeare"  # 단축 URL
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

print(shakespeare_text[:80])
print("글자 수  = %d" % (len(shakespeare_text)))

# tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5)
# tokenizer.fit_on_texts(shakespeare_text)
# print(tokenizer.word_index)

text_vec_layer = tf.keras.layers.TextVectorization(split="character", standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]

encoded -= 2
n_tokens = text_vec_layer.vocabulary_size() - 2
print(text_vec_layer.vocabulary_size())
dataset_size = len(encoded)

print(encoded)

# def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
#     ds = tf.data.Dataset.from_tensor_slices(sequence)
#     ds = ds.window(length + 1, shift=1, drop_remainder=True)
#     ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
#     if shuffle:
#         ds = ds.shuffle(100_000, seed=seed)
#     ds = ds.batch(batch_size)
#     return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


# length = 100
# tf.random.set_seed(42)
# train_set = to_dataset(encoded[:1_000], length=length, shuffle=True, seed=42)
# valid_set = to_dataset(encoded[1_000:1_060], length=length)
# test_set = to_dataset(encoded[1_060:1_200], length=length)

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
#     tf.keras.layers.GRU(128, return_sequences=True),
#     tf.keras.layers.Dense(n_tokens, activation="softmax")
# ])

# model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
# model_ckpt = tf.keras.callbacks.ModelCheckpoint("my_shakespeare_model.keras", monitor="val_accuracy",
#                                                 save_best_only=True)
# history = model.fit(train_set, validation_data=valid_set, epochs=10, callbacks=[model_ckpt])
#
# print(history)
# print(model.evaluate(valid_set))

# shakespeare_model = tf.keras.Sequential([
#     text_vec_layer,
#     tf.keras.layers.Lambda(lambda X: X - 2),
#     model
# ])
#
# y_proba = shakespeare_model.predict(["To be or not to b"])[0, -1]
# y_pred = tf.argmax(y_proba)
# print(text_vec_layer.get_vocabulary()[y_pred + 2])
#
# log_probas = tf.math.log([[0.5, 0.4, 0.1]])
# tf.random.set_seed(42)
# print(log_probas)
# print(tf.random.categorical(log_probas, num_samples=8))
#
# def next_char(text, temperature=1):
#     y_proba = shakespeare_model.predict([text])[0, -1:]
#     rescaled_logits = tf.math.log(y_proba) / temperature
#     char_id = tf.random.categorical(rescaled_logits, num_sample=1)[0, 0]
#     return text_vec_layer.get_vocabulary()[char_id + 2]
#
#
# def extend_text(text, n_chars=50, temperature=1):
#     for _ in range(n_chars):
#         text += next_char(text, temperature)
#     return text
#
#
# tf.random.set_seed(42)
# print(extend_text("To be or not to be", temperature=0.01))
# print(extend_text("To be or not to be", temperature=1))
# print(extend_text("To be or not to be", temperature=100))
