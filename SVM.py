import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel
from sklearn.datasets import load_iris, make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, LinearSVR, SVC

iris = load_iris(as_frame=True)
x = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = (iris.target == 2)

svm_clf = make_pipeline(StandardScaler(), LinearSVC(C=1, random_state=42))
svm_clf.fit(x, y)

X_new = [[5.5, 1.7], [5.0, 1.5]]
print(svm_clf.predict(X_new))
print(svm_clf.decision_function(X_new))

x, y = make_moons(n_samples=100, noise=0.15, random_state=42)

poly_svm_clf = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(),
                             LinearSVC(C=10, max_iter=10_000, random_state=42))

print(poly_svm_clf.fit(x, y))

plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Make Moons Dataset")
plt.show()

poly_kernel_svm_clf = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=3, coef0=1, C=5))
print(poly_kernel_svm_clf.fit(x, y))

# rbf_kernel_svm_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=5, C=0.001))
# rbf_kernel_svm_clf.fit(x, y)
# print(rbf_kernel_svm_clf.score(x, y))

svm_reg = make_pipeline(StandardScaler(), LinearSVR(epsilon=0.5, random_state=42))
svm_reg.fit(x, y)
plt.plot(x[:, 0], x[:, 1], 'b.')
plt.show()
