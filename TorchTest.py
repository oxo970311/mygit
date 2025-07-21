import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from keras.src.utils.module_utils import torchvision
from numpy.ma.core import resize
from tensorflow.python.ops.gen_dataset_ops import tensor_dataset
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import os
import glob

# torch main_name_space 텐서, 수학 등... 함수제공 numpy와 유사
# torch.autograd 자동 미분을 위한 함수들이 포함
# torch.nn 신경망을 구축하기위한 데이터 구조, 레이어 등... 정의
# torch.optim SGD 중심으로 파라미터 최적화 알고리즘 제공
# torch.utils.data SGD 반복 연산 실핼할 때 사용하는 미니 배치용 유틸리티 함수 포함
# torch.onnx 서로 다른 딥러닝 프레임워크간에 모델을 공유할 때 사용하는 포맷
# tensor = tensor 정의, view = 차원 축소, squeeze = 1 인 차원 제거, unsqueeze 1 차원 추가
# torch.cat([0,0],[0,0], dim=n) dim=0,1,2... 차원 방향으로 결합, torch.stack([0,0],[0,0], dim=n) 차원 방향으로 쌓기
# torch.ones.like([x]) tensor_value = 1, torch.zeros.like([x]) tensor_value = 0

# GPU 동작 코드
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# # for reproducibility
# orch.manual_seed(777)
# if device == 'cuda':
#     torch.cuda.manual_seed_all(777)

x_data = torch.tensor([[1, 2], [3, 4]])
y_data = torch.tensor([[7, 8], [9, 10]])
z_data = torch.tensor([[11, 13, 15], [15, 17, 19]])
zeros = torch.tensor([[0, 0], [0, 0]])
prod = x_data @ y_data

print(x_data.shape)
print(prod)

print(torch.__version__)
print(torch.cuda.is_available())

yz = y_data @ z_data

print(yz.shape)

prod.view([-1, 1])

print(yz.shape)

print(yz.unsqueeze(0))
print(yz.unsqueeze(0).shape)

print(torch.cat([prod, zeros], dim=1))

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
print(x_train.shape, y_train.shape)

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=False)

# hypothesis = x_train * w + b
# print(hypothesis)


# optimizer = optim.SGD([w, b], lr=0.01)
# nb_epochs = 300
# for epoch in range(nb_epochs + 1):
#     hypothesis = x_train * w + b
#     mse = torch.mean((hypothesis - y_train) ** 2)
#     optimizer.zero_grad()
#     mse.backward()
#     optimizer.step()
#
#     if mse == 0:
#         break
#
#     else:
#         print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} mse: {:.6f}'.format(
#             epoch, nb_epochs, w.item(), b.item(), mse.item()
#         ))

proto = torch.ones(3, 3, 3)
print(proto, proto.shape)

a = torch.zeros(2, 3)
print(a, a.shape)
a_tf = a.transpose(0, 1)
print(a_tf, a_tf.shape)

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# points[1][0] = 10
second_points = points.clone()
second_points[1][0] = 10
print(points)
print(second_points)

st = torch.ones(5, 4, 6, 7)
print(st.shape)
st = st.transpose(0, 3)
print(st.shape)
st = st.transpose(0, 2)
print(st.shape)

# image_dir = "C:/Users/oxo97/Downloads/dataset/test_set/cats/"
# image_path = glob.glob(os.path.join(image_dir, "*.jpg"))
#
# _list = []
# centercrop = torchvision.transforms.CenterCrop(256)
#
# for path in image_path[:5]:
#     img = cv2.imread(path, 1)
#     image_set = torch.from_numpy(img).permute(2, 0, 1).float()
#     resize_image = centercrop(image_set)
#     _list.append(resize_image)
#     cv2.imshow("cats", img)
#     key = cv2.waitKey(100)
#     cv2.destroyAllWindows()
#
# print("image resize...\n")
# for i, img in enumerate(_list):
#     print(f"이미지 {i} shape: {img.shape}")

# image = cv2.imread("C:/Users/oxo97/Downloads/cat.jpg", 1)
# cv2.imshow('cat', image)
# rs_image = cv2.resize(image,(500,500))
# cv2.imshow('rs_cat', rs_image)
# key = cv2.waitKey(0)
# cv2.destroyAllWindows()


batch_size = 3

batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

print(batch.shape)

# ts_image = torch.from_numpy(image)
# print(ts_image.shape)
# ts_image = ts_image.permute(2, 0, 1)
# print(ts_image.shape)

# Download latest version

wine_data_set = pd.read_csv("C:/Users/oxo97/Downloads/winequality-white.csv", sep=";")
print(wine_data_set.info)

wine_numpy = wine_data_set.to_numpy(dtype='float32')

wine_tensor = torch.from_numpy(wine_numpy)
print(wine_tensor.shape)

wine_data = wine_tensor[:, :-1]
wine_target = wine_tensor[:, -1].long()
print(wine_data, wine_data.shape)
print(wine_target, wine_target.shape)

wine_target_onehot = torch.zeros(wine_target.shape[0], 10)
print(wine_target_onehot, wine_target_onehot.shape)
wine_target_onehot.scatter_(1, wine_target.unsqueeze(1), 1.0)
print(wine_target_onehot, wine_target_onehot.shape)

wine_data_mean = torch.mean(wine_data, dim=0)
wine_data_var = torch.var(wine_data, dim=0)
wine_data_std = torch.sqrt(wine_data_var)

wine_data_normalized = (wine_data - wine_data_mean) / wine_data_std + 1e-6

print(wine_data_normalized, wine_data_normalized.shape)

bad_data = wine_data[wine_target <= 3]
mid_data = wine_data[(wine_target > 3) & (wine_target < 7)]
good_data = wine_data[wine_target >= 7]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

print(bad_mean.shape, mid_mean.shape, good_mean.shape)

for i, args in enumerate(zip(wine_data_set, bad_mean, mid_mean, good_mean)):
    print("{:2} {:24} bad_mean : {:6.2f}, mid_mean : {:6.2f}, good_mean : {:6.2f}".format(i, *args))

total_sulfur_threshold = 141.83
total_sulfur_data = wine_data[:, 6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
print(predicted_indexes, predicted_indexes.shape, predicted_indexes.sum())

actual_indexes = wine_target > 5
print(actual_indexes, actual_indexes.shape, actual_indexes.sum())

n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()

print(n_matches, n_matches / n_predicted, n_matches / n_actual)

# data_set = pd.read_csv("C:/Users/oxo97/Downloads/bike+sharing+dataset/day.csv", sep=";")
# print(data_set.info, data_set.shape)

bikes_numpy = np.loadtxt("C:/Users/oxo97/Downloads/bike+sharing+dataset/hour.csv", dtype="float32", delimiter=",",
                         skiprows=1, converters={1: lambda x: float(x[8:10])})

bikes_tensor = torch.from_numpy(bikes_numpy)
print(bikes_tensor.shape, bikes_tensor.stride())
print(bikes_tensor)

bikes_tensor = bikes_tensor[:17376]
bikes_tensor.reshape((17376, 17))
print(bikes_tensor.shape)

daily_bikes = bikes_tensor.view(-1, 24, bikes_tensor.shape[1])
print(daily_bikes.shape, daily_bikes.stride())

daily_bikes = daily_bikes.transpose(1, 2)
print(daily_bikes.shape, daily_bikes.stride())

first_day = bikes_tensor[:24].long()
weather_onehot = torch.zeros(first_day.shape[0], 4)
print("first_day")
print(first_day.shape)
print(first_day[:, 9])

weather_onehot.scatter_(
    dim=1,
    index=first_day[:, 9].unsqueeze(1).long() - 1,
    value=1.0
)

print(weather_onehot.shape)
print(weather_onehot)

cat_bikes = torch.cat((bikes_tensor[:24], weather_onehot), 1)[:1]
print("cat_bikes")
print(cat_bikes)

daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
print(daily_weather_onehot)

daily_weather_onehot.scatter_(
    1, daily_bikes[:, 9, :].long().unsqueeze(1) - 1, 1.0)

print(daily_weather_onehot.shape)
print(daily_weather_onehot)

daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), 1)
print(daily_bikes)

daily_bikes[:, 9, :] = (daily_bikes[:, 9, :] - 1.0) / 3.0

temp = daily_bikes[:, 10, :]
temp_min = torch.min(temp)
temp_max = torch.max(temp)
daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - temp_min / temp_max - temp_min))

print(daily_bikes[:, 10, :])
# daily_bikes[:, 10, :] = ((daily_bikes[:, 10, :] - torch.mean(temp) / torch.std(temp)))
