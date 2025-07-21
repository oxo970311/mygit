import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans

url = "https://bit.ly/fruits_300_data"
response = requests.get(url)
with open('fruits_300.npy', 'wb') as file:
    file.write(response.content)

fruits = np.load('fruits_300.npy')

fruits_2d = fruits.reshape(-1, 100 * 100)
print(fruits)

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
print(km.labels_)

print(np.unique(km.labels_, return_counts=True))

count = 1
def draw_fruits(arr, radio=1):
    global count
    n = len(arr)
    print("labels_%d 과일 갯수 : %d" % (count, n))
    count += 1
    rows = int(np.ceil(n / 10))

    cols = n if rows < 2 else 10

    fig, axs = plt.subplots(rows, cols, figsize=(rows * radio, cols * radio), squeeze=False)

    for i in range(rows):
        for j in range(cols):

            if i * 10 + j < n:
                axs[i, j].imshow(arr[i * 10 + j], cmap="gray_r")
            axs[i, j].axis('off')
    plt.show()


draw_fruits(fruits[km.labels_ == 0])
draw_fruits(fruits[km.labels_ == 1])
draw_fruits(fruits[km.labels_ == 2])

draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), radio=3)
