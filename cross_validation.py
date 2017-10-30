from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt

# get the data, handle the data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# draw image
# plt.subplot(121)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

# knn = KNeighborsClassifier(n_neighbors=5)

############ knn without cross_validation
# knn.fit(X_train, y_train)

# print(knn.score(X_test, y_test))

########### knn with cross_validation
# score = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
# print(score.mean())

# draw image
# plt.subplot(122)
# prediction = knn.predict(X_test)
# plt.scatter(X_test[:, 0], X_test[:, 1], c=prediction)
# plt.show()

####################### find out which n_neighbors is the best
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error') # for regression
    # score = cross_val_score(knn, X, y, cv=5, scoring='accuracy') # for classification

    # k_scores.append(score.mean())
    k_scores.append(loss.mean())

plt.plot(k_range, k_scores, 'r-')
plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validation Accuracy')
plt.ylabel('Cross-Validation Error')
plt.show()