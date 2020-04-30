#!/usr/bin/env python
# encoding: utf-8
'''
@author: wty
@license: (C) Copyright 2019-2020, Node Supply Chain Manager Corporation Limited.
@contact: wty229027377@gmail.com
@software: pycharm
@file: buston_home_price.py
@time: 2020/4/1 23:28
@desc:
'''
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib as mpl
import matplotlib.pyplot as plt
import threading
from LinearRegression import LinearRegression


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
    mpl.rcParams['font.sans-serif'] = [u'simHei']
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


def createModel(train_data, train_result, iter=30000):
    col = np.shape(train_data)[1] + 1
    theta = np.random.random((col, 1))
    lr_BGD = LinearRegression(train_data, train_result, theta)
    lr_SGD = LinearRegression(train_data, train_result, theta)
    lr_MBGD = LinearRegression(train_data, train_result, theta)
    lr_NormalEquation = LinearRegression(train_data, train_result, theta)

    alpha = 0.001
    batch_size = 64
    ticks = time.time()
    print(ticks)
    # bgd_t1 = MyThread(lr_BGD.train_BGD, (iter, alpha))
    # sgd_t2 = MyThread(lr_SGD.train_SGD, (iter, alpha))
    # mbgd_t3 = MyThread(lr_MBGD.train_MBGD, (iter, batch_size, alpha))
    # bgd_t1.start()
    # sgd_t2.start()
    # mbgd_t3.start()
    # bgd_t1.join()
    # sgd_t2.join()
    # mbgd_t3.join()
    #
    # BGD_train_cost = bgd_t1.get_result()
    # SGD_train_cost = sgd_t2.get_result()
    # MBGD_train_cost = mbgd_t3.get_result()
    """
    多进程
    """
    jobs = []
    P = Pool(3)
    BGD_train_cost = P.map(lr_BGD.train_BGD, {iter, alpha})

    SGD_train_cost = P.map(lr_SGD.train_SGD, [iter, alpha])

    MBGD_train_cost = P.map(lr_MBGD.train_MBGD, [iter, batch_size, alpha])
    P.close()
    P.join()


    # BGD_train_cost = lr_BGD.train_BGD(iter, alpha)
    # SGD_train_cost = lr_SGD.train_SGD(iter, alpha)
    # MBGD_train_cost = lr_MBGD.train_MBGD(iter, batch_size, alpha)
    ticks2 = time.time()
    print(ticks2 - ticks)
    # lr_NE = lr_NormalEquation.getNormalEquation()

    return BGD_train_cost, SGD_train_cost, MBGD_train_cost  # , lr_NE


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


if __name__ == '__main__':
    print("哈哈哈 开始了 %d", time.time())
    train_data, test_data, train_result, test_result = handleData()

    showPhoto(test_data, test_result)
    ticks = time.time()
    BGD_train_cost, SGD_train_cost, MBGD_train_cost = createModel(train_data, train_result)
    print(time.time() - ticks)

    ticks = time.time()
    costView(BGD_train_cost, SGD_train_cost, MBGD_train_cost)
    print(time.time() - ticks)
    print("wo艹 结束了 %d", time.time())
    # 非多线程下一共114秒
