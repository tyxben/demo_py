#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/4/30 下午8:14
# @Author  : wty
# @Email   : wty229027377@gmail.com
# @File    : k-cross-validation.py
# @Software: PyCharm

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
X, y = iris.data, iris.target

paramers = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, paramers, cv=5)
clf.fit(X, y)
print("best_score%.2f" % clf.best_score_, "best_k:", clf.best_params_)
