#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/4/30 下午7:46
# @Author  : wty
# @Email   : wty229027377@gmail.com
# @File    : Cross-validation.py
# @Software: PyCharm

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# 加载 iris
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X.shape, y.shape)

# 定义要交叉验证的K值选集
ks = [1, 3, 5, 7, 9, 11, 13, 15]

kf = KFold(n_splits=5, random_state=2001, shuffle=True)

# 保存当前最好的K值和对应的准确率
best_k = ks[0]
best_score = 0

# 循环每一个K 值
for k in ks:
    curr_score = 0
    for train_index, valid_index in kf.split(X):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X[train_index], y[train_index])
        curr_score = curr_score + clf.score(X[valid_index], y[valid_index])
    avg_score = curr_score / 5
    if avg_score > best_score:
        best_k = k
        best_score = avg_score
    print("current best score is:%.2f" % best_score, "best k:%d" % best_k)
print("after cross validation, the final best k is:%d" % best_k)
