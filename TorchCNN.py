import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import optim
import os
import glob
import cv2

path_ = glob.glob("C:/Users/oxo97/Downloads/dataset/training_set/dogs/*.jpg")
target_path_ = glob.glob("C:/Users/oxo97/Downloads/dataset/test_set/dogs/*.jpg")


for path in path_[:1000]:
    img = Image.open(path)
    resize = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))
    ])
    re_img = resize(img)
    # img_to_show = re_img.permute(1, 2, 0)
    # plt.imshow(img_to_show)
    # plt.show()
    # print(re_img.shape)
    # CHW_img = re_img.permute(2, 0, 1)
    # print(CHW_img.shape)

for target_path in target_path_[:200]:
    target_img = Image.open(target_path)
    resize = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))
    ])
    re_target_img = resize(target_img)
    # target_img_to_show = re_target_img.permute(1, 2, 0)
    # plt.imshow(target_img_to_show)
    # plt.show()

# class new_model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(10,20)
#         self.linear2 = nn.Linear(20, 40)
#         self.linear3 = nn.Linear(40, 20)
#         self.linear4 = nn.Linear(20, 1)
#         self.relu = nn.ReLU()
#         # self.softamx = nn.Softmax()
#
#
#     def forward(self,x):
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)
#         x = self.relu(x)
#         x = self.linear3(x)
#         x = self.relu(x)
#         x = self.linear4(x)
#
#         return x

print(re_img.shape)
print(re_target_img.shape)
# print(img_to_show.shape)
# print(target_img_to_show.shape)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 128 * 128, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

re_img = re_img.unsqueeze(0)

model = CNNModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)



epochs = 1000
for epoch in range(epochs):
    output = model(re_img)
    # target = torch.tensor([[1.0]])
    target = re_target_img[:200]

    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    if loss.item() < 1e-5:
        break

