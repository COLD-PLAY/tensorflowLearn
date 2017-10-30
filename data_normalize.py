from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np

# a = np.array([[10, 2.7, 3.6],
#               [-100, 5, -2],
#               [120, 20, 40]], dtype=np.float64)
# print(a)
# print(preprocessing.scale(a))

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                            random_state=22, n_clusters_per_class=1, scale=100)

# plt.subplot(121)
# plt.scatter(X[:, 0], X[:, 1], c=y)

X = preprocessing.scale(X)

# plt.subplot(122)
# plt.scatter(X[:, 0], X[:, 1], c=y)

# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC()

clf.fit(X_train, y_train)
# clf.fit(X, y)
prediction = clf.predict(X)
# prediction = clf.predict(X_test)

plt.subplot(121)
plt.title('本来的分类')
plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

plt.subplot(122)
plt.title('预测的分类')
plt.scatter(X[:, 0], X[:, 1], c=prediction)
# plt.scatter(X_test[:, 0], X_test[:, 1], c=prediction)

plt.show()

# print(clf.score(X_test, y_test))
print(clf.score(X, y))