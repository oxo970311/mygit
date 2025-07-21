import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, add_dummy_feature

# df = pd.read_csv('https://bit.ly/perch_csv_data')
# perch_full = df.to_numpy()
# print(perch_full)
#
# perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
#                          115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
#                          150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
#                          218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
#                          556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
#                          850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
#                          1000.0])
#
# train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
#
# poly = PolynomialFeatures(include_bias=False)
# # poly.fit([[2, 3]])
# # print(poly.transform([[2, 3]]))
# poly.fit(train_input)
# train_poly = poly.transform(train_input)
# print(train_poly.shape)
# print(poly.get_feature_names_out())
#
# test_poly = poly.transform(test_input)
#
# lr = LinearRegression()
# lr.fit(train_poly, train_target)
# print(lr.score(train_poly, train_target))
# print(lr.score(test_poly, test_target))
#
# poly = PolynomialFeatures(degree=5, include_bias=False)
# poly.fit(train_input)
# train_poly = poly.transform((train_input))
# test_poly = poly.transform((test_input))
# print(train_poly.shape)
# print(poly.get_feature_names_out())
#
# lr = LinearRegression()
# lr.fit(train_poly, train_target)
# print(lr.score(train_poly, train_target))
# print(lr.score(test_poly, test_target))
#
# ss = StandardScaler()
# ss.fit(train_poly)
# train_scaled = ss.transform(train_poly)
# test_scaled = ss.transform((test_poly))
#
# ridge = Ridge()
# ridge.fit(train_scaled, train_target)
# print(ridge.score(train_scaled, train_target))
# print(ridge.score(test_scaled, test_target))
#
# train_score = []
# test_score = []
# alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
# for alpha in alpha_list:
#     ridge = Ridge(alpha=alpha)
#     ridge.fit(train_scaled, train_target)
#     train_score.append(ridge.score(train_scaled, train_target))
#     test_score.append(ridge.score(test_scaled, test_target))
#
# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.show()

np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

print(X[:5])
print(y[:5])
plt.plot(X, y, 'b.')
plt.show()

X_b = add_dummy_feature(X)
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(theta_best)

X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new)
y_predcit = X_new_b @ theta_best
print(y_predcit)

plt.plot(X_new, y_predcit, '-r', label='예측')
plt.plot(X, y, 'b.')
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
print(lin_reg.predict(X_new))

eta = 0.1
n_epochs = 1000

np.random.seed(42)
theta = np.random.randn(2, 1)

for epochs in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

print(theta)

plt.plot(X, y, 'b.')
plt.plot(X_new, gradients)
plt.show()
