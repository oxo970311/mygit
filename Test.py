import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import glob
import os
from pathlib import Path
import math
from PIL import Image
import random

from numpy.matlib import empty
from tensorflow.python.ops.gen_batch_ops import batch

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# path = Path("C:/Users/oxo97/Downloads/CTA_-_Ridership_-_Daily_Boarding_Totals_20250309.csv")
# df = pd.read_csv(path, parse_dates=["service_date"])
# df.columns = ["date", "day_type", "bus", "rail", "total"]
# df = df.sort_values("date").set_index("date")
# df = df.drop("total", axis=1)
# print(df.info)
#
# arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# df_arr = pd.DataFrame(arr)
# df_arr.loc[-1:] += 2
# print(df_arr)
#
# my_series = [0, 1, 2, 3, 4, 5]
# my_dataset = tf.keras.utils.timeseries_dataset_from_array(
#     my_series,
#     targets=my_series[3:],
#     sequence_length=3,
#     batch_size=2
# )
#
# print(list(my_dataset))
# # print(df["rail"]["2016-01":"2018-12"])
# rail_train = df["rail"]["2016-01":"2018-12"] / 1e6
# rail_vaild = df["rail"]["2019-01":"2019-05"] / 1e6
# rail_test = df["rail"]["2019-06":] / 1e6
#
# rt_arr = rail_train.to_numpy()
# print(rt_arr)
#

# dataset = tf.data.Dataset.range(10).repeat(4)
# dataset = dataset.shuffle(buffer_size=4, seed=42).batch(7)
# for item in dataset:
#     print(item)
#
#
# class MyNormalization(tf.keras.layers.Layer):
#     def adapt(self, X):
#         self.mean_ = np.mean(X, axis=0, keepdims=True)
#         self.std_ = np.std(X, axis=0, keepdims=True)
#
#     def call(self, inputs):
#         eps = tf.keras.backend.epsilon()
#         return (inputs - self.mean_) / (self.std_ + eps)


# norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[:1])
# model = tf.keras.Sequential([
#     norm_layer,
#     tf.keras.layers.Dense(50, activation="relu"),
#     tf.keras.layers.Dense(50, activation="relu"),
#     tf.keras.layers.Dense(50, activation="relu"),
#     tf.keras.layers.Dense(1, activation="softmax")
# ])

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# image_paths = glob.glob("C:/Users/oxo97/.cache/kagglehub/datasets/muratkokludataset/rice-image-dataset/versions/1/Rice_Image_Dataset/Arborio/*.jpg")
# image_size = (250,250)
#
# images = []
# for path in image_paths:
#     img = Image.open(path).convert("RGB")
#     img = img.resize(image_size)
#     img = np.array(img)
#     images.append(img)

# fig, ax = plt.subplots(1,10)
# for i in range (10):
#     ax[0].imshow(images[i])
#     plt.axis('off')
#     plt.show()
#
# plt.imshow(images[1])
# plt.show()

# import torch
# print(torch.__version__)
# print(tf.__version__)
#
# x = torch.rand(5, 3)
# print(x)

student_mid = np.array([70, 85, 90, 75])
student_fin = np.array([90, 65, 70, 85])
student_sum = student_mid + student_fin
print(student_sum / 2)


# try:
#     a, b = input("두 수를 입력하시오 : ").split()
#     result = int(a) / int(b)
#     print('{} / {}'.format(a, b, result))
#
# except:
#     print("수가 정확한지 확인하세요.")
#
# l = []
# t = ()
# try:
#     # b = 2 / 0
#     # a = 1 + "hundred"
#     c = 7 + l[0]
#
# except Exception as e:
#     print("error :", e)


class Person:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def _information_(self):
        print("이름 : %s" % (self.name))
        print("나이 : %d" % (self.age))

    def _age_(self):
        if self.age >= 30:
            print("30 이상.")

        else:
            print("30 이하.")


p = Person("yoo", 29)
p._information_()
p._age_()


class Cat(Person):
    def meow(self):
        print("meow meow")


nabi = Cat("nabi", 5)
nabi.meow()
nabi._information_()

print(dir(int))
print(dir(list))


# class Manager(Person):
#     def __init__(self, name, position):
#         super().__init__(self, name)
#         self.position = position

def square(x):
    return x ** 2


a = [1, 2, 3, 4, 5, 6, 7]

square_a = list(map(square, a))
print(square_a)

b = [x ** 2 for x in range(1, 8) if x % 2 == 0]
print(b)


def square(a):
    b = a * a
    return math.sqrt(b)


value = square(5)
print(value)

list = []
temp = 0
for idx in range(20):
    value = random.randrange(1, 100)
    list.append(value)
    if temp < list[idx]:
        temp = list[idx]
        temp_idx = idx

print("Max Value : ", temp)
print("Max Value index : ", temp_idx)
print("list : ", list)
print("list len : %d" % (len(list)))

name = ["Tom", "Jerry", "Mike", "elsa", "Tom"]
k = 0
for i in range(k, len(name)):
    for j in range(i, 5):
        if name[i] == name[j]:
            print("Tom Duplication!")
            break

        else:
            print(name[i])
    k += 1

name_set = ["Tom", "jerry", "Mike"]
couple = []

for i in range(0, len(name_set) - 1):
    coup = name_set[i] + name_set[i + 1]
    couple.append(coup)
    print(couple)

# def factorial(a):
#     for i in range(a, 0, -1):
#         if i <= 1:
#             return 1
#
#         else:
#             return factorial(i) * factorial(i - 1)


# print(factorial(5))
#
# def fac(a):
#     v = 1
#     for i in range(a, 0, -1):
#         v *= i
#         if i == 1:
#             return v
#
# value = fac(5)
# print(value)

# for i in range(5, 0 ,-1):
#     print(i)

# pillar1 = [1, 2, 3]
# pillar2 = []
# pillar3 = []
#
#
# def hanoi(pillar1):
#     i = 0
#     while (len(pillar3) == 3):
#
#         if len(pillar2) == 0:
#             pillar2[i] = pillar1[i]
#             i += 1
#
#         elif len(pillar3) == 0:
#             pillar3[i - 1] = pillar1[i]
#             i -= 1
#
#         elif len(pillar1) == 1:
#             pillar3[i + 1] = pillar1[i + 2]
#
#         pillar3[i + 2] = pillar2[i]
#
#     print(pillar1, pillar2, pillar3)
#
#
# hanoi(pillar1)

choice = int(input("choice number : "))

list = []
for i in range(0,10):
    list.append(random.randrange(1,10))

def select_number(list, n):
    for idx in range(0,10):
        if list[idx] == n:
            return idx


value = select_number(list, choice)
print(value)
print(list)