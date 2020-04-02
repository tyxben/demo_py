#!/usr/bin/env python
# encoding: utf-8
'''
@author: wty
@license: (C) Copyright 2019-2020, Node Supply Chain Manager Corporation Limited.
@contact: wty229027377@gmail.com
@software: pycharm
@file: numpy_do.py
@time: 2020/3/13 0:37
@desc:
'''

import numpy as np
from scipy import linalg


class Vector:
    def __init__(self):
        print("test")

    def Sum(self, v1, v2):
        return v1 + v2

    def Mul(self, v2, v1=1):
        return v1 * v2


if __name__ == '__main__':
    #  行列式
    a = np.array([1, 2, 3, 4])
    print(a)

    # 转置
    # method 1
    '''np.transpose 一维数组无效'''
    A_t = a[:, np.newaxis]
    print(A_t)

    # method 2
    a2 = np.array([[1, 2, 3, 4]])
    A_t1 = a2.T
    print(A_t1)

    ## 向量加法

    vec = Vector()
    result = vec.Sum(A_t1, A_t)
    print(result)

    result_mul = vec.Mul(result, 2)
    print(result_mul)

    i_eye = np.eye(5)
    print(i_eye)

    A_liang = np.array([[1, 35, 0], [0, 2, 3], [0, 0, 4]])
    A_n = linalg.inv(A_liang)
    print(A_n)
    print(np.dot(A_liang, A_n))

    AAA = np.array([[1, 2, 3], [1, -1, 4], [2, 3, -1]])
    y = np.array([14, 11, 5])
    x = linalg.solve(AAA, y)
    print(x)

