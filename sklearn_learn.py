import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.image as mpimg

####################### this is an example for KNeigghborsClassifier to classify the iris(flowers)
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# print(iris_X[:2, :])
# print(iris_y)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
# print(y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

prediction = knn.predict(X_test)
print(prediction)
print(y_test)

err = 0
num = len(y_test)
for i in range(num):
    if prediction[i] != y_test[i]:
        err += 1

print('error rate: ', str(err / num))