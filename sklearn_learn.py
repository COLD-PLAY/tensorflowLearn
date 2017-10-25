import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.image as mpimg

# iris = datasets.load_iris()
# iris_X = iris.data
# iris_y = iris.target

faces = datasets.fetch_olivetti_faces()
faces_data = faces.data

houses = datasets.fetch_california_housing()
houses_data = houses.data

# print(iris_X[:2, :])
# print(iris_y)

# print(houses_data.shape)
# print(houses_data[:2, :])
# print(houses.image)


# for i in range(len(houses_data)):
#     print(i)
#     filename = 'sklearn_data/houses/' + str(i) + '.jpg'
#     mpimg.imsave(filename, houses_data[i, :].reshape(64, 64))