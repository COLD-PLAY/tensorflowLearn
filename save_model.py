from sklearn import svm
from sklearn import datasets

# clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
# clf.fit(X, y)

######## method1: pickle
# import pickle

#### save
# with open('save/clf.pickle', 'wb') as fr:
#     pickle.dump(clf, fr)

#### load
# with open('save/clf.pickle', 'rb') as fr:
#     clf2 = pickle.load(fr)

# print(clf2.score(X, y))

######## method2: joblib
from sklearn.externals import joblib

#### save
# joblib.dump(clf, 'save/clf.pkl')

#### load
clf3 = joblib.load('save/clf.pkl')
print(clf3.score(X, y))