
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import joblib

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
clf = KNeighborsClassifier()
clf.fit(iris_x,iris_y)

joblib.dump(clf, 'knn.pkl')