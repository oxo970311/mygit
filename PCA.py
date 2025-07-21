import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

mnist = fetch_openml('mnist_784', as_frame=False)

x_train, x_test, y_train, y_test = train_test_split(mnist.data[:2000], mnist.target[:2000], test_size=0.2,
                                                    random_state=42)

pca = PCA()
pca.fit(x_train)

X_centered = x_train - x_train.mean(axis=0)

U, s, Vt = np.linalg.svd(X_centered)

c1 = Vt[0]
c2 = Vt[1]

print(c1, c2)

W2 = Vt[:2].T
X2D = X_centered @ W2
print(X2D)
# print(pca.explained_variance_ratio_)

cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print(d)

pca = PCA(n_components=0.95)
x_reduced = pca.fit_transform(x_train)
print(x_reduced)
print(pca.n_components_)

clf = make_pipeline(PCA(random_state=42), RandomForestClassifier(random_state=42))

param_distrib = {"pca__n_components": np.arange(10, 80),
                 "randomforestclassifier__n_estimators": np.arange(50, 500)}

rnd_search = RandomizedSearchCV(clf, param_distrib, n_iter=10, cv=3, random_state=42)
rnd_search.fit(x_train[:1000], y_train[:1000])
print(rnd_search.best_params_)
