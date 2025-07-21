import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data[:2000], mnist.target[:2000]


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


some_digit = X[10]
plot_digit(some_digit)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_clf = SVC(random_state=42)
print(svm_clf.fit(X_train, y_train))
print(svm_clf.predict([some_digit]))

some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores.round(2))

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train, y_train)

ovr_clf.predict([some_digit])
print(len(ovr_clf.estimators_))

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])
print(sgd_clf.decision_function([some_digit]).round())
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy'))
print(sgd_clf.score(X_train, y_train))

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy'))

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
print(y_train_pred)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, normalize="true", values_format=".0%")
plt.show()

sample_weight = (y_train_pred != y_train)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, sample_weight=sample_weight, normalize='true',
                                        values_format='.0%')
plt.show()

y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multiabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multiabel)
