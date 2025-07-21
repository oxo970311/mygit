import matplotlib.pyplot as plt
import requests
import numpy as np
import wget

url = "https://bit.ly/fruits_300_data"
response = requests.get(url)

# 다운로드한 내용을 파일로 저장
with open('fruits_300.npy', 'wb') as file:
    file.write(response.content)

fruits = np.load('fruits_300.npy')
print(fruits.shape)

print(fruits[0, 0, :])

plt.imshow(fruits[0], cmap='gray')
plt.show()

fig, axs = plt.subplots(1, 3)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
axs[2].imshow(fruits[299], cmap='gray_r')
plt.show()

apple = fruits[0:100].reshape(-1, 100 * 100)
pineapple = fruits[100:200].reshape(-1, 100 * 100)
banana = fruits[200:300].reshape(-1, 100 * 100)

apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)
fit, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)

apple_index = np.argsort(abs_mean)[:100]
print(apple_index)
fig,axs = plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i,j].imshow(fruits[apple_index[i*10+j]],cmap="gray_r")
        axs[i,j].axis('off')
plt.show()