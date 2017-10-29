from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

# loaded_data = datasets.load_boston()
# data_X = loaded_data.data
# data_y = loaded_data.target

# model = LinearRegression()
# model.fit(data_X, data_y)

# print(model.predict(data_X[:4]))
# print(data_y[:4])

# X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10.)
# plt.scatter(X, y)

# model = LinearRegression()
# model.fit(X, y)

# print(model.coef_, model.intercept_) # y = W*X + b

# # plt.scatter(X, model.coef_ * X + model.intercept_)
# plt.plot(X, model.coef_ * X + model.intercept_, 'go')
# plt.show()

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.3)

# plt.plot(data_X, data_y, 'go')
# plt.scatter(data_X, data_y)
# plt.show()

model = LinearRegression()
# model.fit(data_X, data_y)
model.fit(train_X, train_y)

# print(model.coef_)
# print(model.intercept_)

# prediction = model.predict([data_X[0]])
# print(prediction, data_y[0])

print(model.get_params())
# print(model.score(data_X, data_y))
print(model.score(test_X, test_y))