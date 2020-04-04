#!/usr/bin/env python
# encoding: utf-8
"""
@author: wty
@license: (C) Copyright 2019-2020, Node Supply Chain Manager Corporation Limited.
@contact: wty229027377@gmail.com
@software: pycharm
@file: buston_home_price.py
@time: 2020/4/1 23:28
@desc:
"""
import time
from pathos.multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib as mpl
import matplotlib.pyplot as plt
import threading
import itertools
from LinearRegression import LinearRegression
from functools import partial


class MyThread(threading.Thread):

    def __init__(self, func, args):
        super(MyThread, self).__init__()

        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def Merge(data, col):
    data = np.array(data).T
    return pd.DataFrame(data, columns=col)


def handleData():
    inputData, resultData = load_boston(return_X_y=True)
    inputData = np.array(inputData)[:, 5]
    data = Merge([inputData, resultData], ['平均房间数目', '房价'])
    data.to_excel(r'../originalData_buston.xlsx')
    inputData = inputData.reshape((len(inputData), 1))
    resultData = np.array(resultData).reshape((len(resultData), 1))

    return train_test_split(inputData, resultData, test_size=0.1, random_state=50)


def showPhoto(test_data, test_result):
    # 用来正常显示中文标签，SimHei是字体名称，字体必须再系统中存在
    plt.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    col = ['真实房价']
    plt.scatter(test_data, test_result, alpha=0.5, c='b', s=10)
    plt.grid(True)
    plt.legend(labels=col, loc='best')
    plt.xlabel("房间数")
    plt.ylabel("真实房价")
    plt.savefig("../test_data_view.jpg", bbox_inches='tight')
    plt.show()
    plt.close()


def executor(tpl):
    return tpl[0](tpl[1])


def createModel(train_data, train_result, lr, iter=30000):
    """
    多进程
    """
    p = Pool(4)
    pre_data = [(lr[0].train_BGD, (iter, alpha)), (lr[1].train_SGD, (iter, alpha)),
                (lr[2].train_MBGD, (iter, batch_size, alpha)), (lr[3].getNormalEquation, (iter, alpha))]
    result = p.map(executor, pre_data)


    return result[0], result[1], result[2], result[3]


def costView(BGD_train_cost, SGD_train_cost, MBGD_train_cost, iter=30000):
    col = ['BGD', 'SGD', 'MBGD']
    iter = np.arange(iter)
    plt.plot(iter, BGD_train_cost, '-r')
    plt.plot(iter, SGD_train_cost, '-b')
    plt.plot(iter, MBGD_train_cost, '-k')
    plt.grid(True)
    plt.xlabel('迭代次数')
    plt.ylabel('平均训练损失')
    plt.legend(labels=col, loc='best')

    plt.savefig("../train_cost_picture_buston.jpg", bbox_inches='tight')
    plt.show()
    plt.close()
    train_cost = [BGD_train_cost, SGD_train_cost, MBGD_train_cost]
    train_cost = Merge(train_cost, col)
    train_cost.to_excel("../train_cost_buston.xlsx")
    train_cost.describe().to_excel("../BGD_SGD_MBGD_cost_average.xlsx")


def testView(test_data, test_result, lr):
    x = np.arange(int(np.min(test_data)), int(np.max(test_data) + 1))
    x = x.reshape((len(x), 1))
    BGD = lr[0].test(x)
    SGD = lr[1].test(x)
    MBGD = lr[2].test(x)
    NE = lr[3].test(x)

    col = ['BGD', 'SGD', 'MBGD', 'NE']
    plt.plot(x, BGD, 'r-.')
    plt.plot(x, SGD, 'b-')
    plt.plot(x, MBGD, 'k--')
    plt.plot(x, NE, 'g:',)
    plt.scatter(test_data, test_result, alpha=0.5, c='b', s=10)
    plt.grid(True)
    plt.xlabel("房间数")
    plt.ylabel("预测值")
    plt.legend(labels=col, loc='best')
    plt.savefig("./predict.jpg", bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    print("哈哈哈 开始了 %d", time.time())
    train_data, test_data, train_result, test_result = handleData()

    showPhoto(test_data, test_result)
    col = np.shape(train_data)[1] + 1
    theta = np.random.random((col, 1))
    alpha = 0.001
    batch_size = 64

    lr_BGD = LinearRegression(train_data, train_result, theta)
    lr_SGD = LinearRegression(train_data, train_result, theta)
    lr_MBGD = LinearRegression(train_data, train_result, theta)
    lr_NormalEquation = LinearRegression(train_data, train_result, theta)
    list_lr = [lr_BGD, lr_SGD, lr_MBGD, lr_NormalEquation]
    BGD_train_cost, SGD_train_cost, MBGD_train_cost, NE = createModel(train_data, train_result, list_lr)

    costView(BGD_train_cost, SGD_train_cost, MBGD_train_cost)
    testView(test_data, test_result, list_lr)
