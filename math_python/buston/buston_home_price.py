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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib as mpl
import matplotlib.pyplot as plt
import threading


class LinearRegression(object):

    def __init__(self, input_data, real_result, theta=None):
        """
        :param input_data: 输入数据
        :param real_result: 真实结果
        :param theta: 线性回归参数，默认为None
        :return: 无
        """
        row, col = np.shape(input_data)

        # 构造输入数据
        self.Input_data = [0] * row

        # 给输入的每个数据添加常数项1
        for (index, data) in enumerate(input_data):
            Data = [1.0]
            Data.extend(list(data))
            self.Input_data[index] = Data
        self.Input_data = np.array(self.Input_data)
        self.Result = real_result

        if theta is not None:
            self.Theta = theta
        else:
            self.Theta = np.random.normal((col + 1, 1))

    def Cost(self):
        """
        计算损失函数
        :return: cost 损失值
        """
        # 损失函数定义: 线性回归中真是结果与预测值之间的均方误差
        predict = self.Input_data.dot(self.Theta).T
        cost = predict - self.Result.T
        cost = np.average(cost ** 2)
        return cost

    def predict(self, data):
        """
        对数据进行预测
        :param data: 测试数据
        :return: 预测结果
        """
        tmp = [1, 0]
        tmp.extend(data)
        data = np.array(tmp)
        return data.dot(self.Theta)[0]

    def test(self, test_data):
        """
        对测试数据集做线性回归预测
        :param test_data: 测试数据
        :return: 预测结果
        """
        predict_result = []
        for data in test_data:
            predict_result.append(self.predict(data))

        return np.array(predict_result)

    def Shuffle_Sequence(self):
        """
        随机打乱数据集
        :return: 返回打乱数据集的序列
        """
        length = len(self.Input_data)
        random_sequence = list(range(length))
        random_sequence = np.random.permutation(random_sequence)
        return random_sequence

    def BGD(self, alpha):
        """
        BGD 算法的具体实现
        :param alpha: 学习率
        :return:
        """
        gradient_increasment = []
        for (input_data, real_result) in zip(self.Input_data, self.Result):
            # 计算每一组 input_data 的增量，并放入递增数组
            g = (real_result - input_data.dot(self.Theta)) * input_data
            gradient_increasment.append(g)
        avg_g = np.average(gradient_increasment, 0)
        avg_g = avg_g.reshape((len(avg_g), 1))
        self.Theta = self.Theta + alpha * avg_g

    def SGD(self, alpha):
        """
        SGD 进行一次迭代调整参数
        :param alpha: 学习速率
        :return:
        """
        shuffle_sequence = self.Shuffle_Sequence()
        self.Input_data = self.Input_data[shuffle_sequence]
        self.Result = self.Result[shuffle_sequence]
        for (input_data, real_result) in zip(self.Input_data, self.Result):
            g = (real_result - input_data.dot(self.Theta)) * input_data
            g = g.reshape((len(g), 1))
            self.Theta = self.Theta + alpha * g

    def MBGD(self, alpha, batch_size):
        """
        MGBD 小批量算法进行迭代
        :param alpha: 学习速率
        :param batch_size: 小批量样本规模
        :return:
        """
        shuffle_sequence = self.Shuffle_Sequence()
        self.Input_data = self.Input_data[shuffle_sequence]
        self.Result = self.Result[shuffle_sequence]
        for start in np.arange(0, len(shuffle_sequence), batch_size):
            end = np.min([start + batch_size, len(shuffle_sequence)])
            mini_batch = shuffle_sequence[start:end]
            mini_train_data = self.Input_data[mini_batch]
            mini_train_result = self.Result[mini_batch]
            gradient_increasment = []
            for (data, result) in zip(mini_train_data, mini_train_result):
                g = (result - data.dot(self.Theta)) * data
                gradient_increasment.append(g)
            avg_g = np.average(gradient_increasment,0)
            avg_g = avg_g.reshape((len(avg_g), 1))
            self.Theta = self.Theta + alpha * avg_g

    def getNormalEquation(self):
        """
        利用正则方程计算模型参数 self.Theta
        :return:
        """
        col, rol = np.reshape(self.Input_data.T)
        xt = self.Input_data.T + 0.001 * np.eye(col, rol)
        inv = np.linalg.inv(xt.dot(self.Input_data))
        self.Theta = inv.dot(xt.dot(self.Result))

    def train_BGD(self, iter, alpha):
        """
        利用BGD 迭代优化函数
        :param iter: 迭代次数
        :param alpha: 学习速率
        :return: 损失数组
        """
        cost = []
        for i in range(iter):
            self.BGD(alpha)
            cost.append(self.Cost())
        cost = np.array(cost)
        return cost

    def train_SGD(self, iter, alpha):
        """
        SGD 迭代
        :param iter: 迭代次数
        :param alpha: 学习速率
        :return: 损失数组
        """
        cost = []
        for i in range(iter):
            self.SGD(alpha)
            cost.append(self.Cost())
        cost = np.array(cost)
        return cost

    def train_MBGD(self, iter, mini_batch, alpha):
        """
        MBGD 迭代
        :param iter: 迭代次数
        :param mini_batch: 小样本批量规模
        :param alpha: 学习速率
        :return: 损失数组
        """
        cost = []
        for i in range(iter):
            self.MBGD(alpha, mini_batch)
            cost.append(self.Cost())
        cost = np.array(cost)
        return cost


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

    BGD_train_cost = lr_BGD.train_BGD(iter, alpha)
    SGD_train_cost = lr_SGD.train_SGD(iter, alpha)
    MBGD_train_cost = lr_MBGD.train_MBGD(iter, batch_size, alpha)
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
    train_data, test_data, train_result, test_result = handleData()

    showPhoto(test_data, test_result)
    BGD_train_cost, SGD_train_cost, MBGD_train_cost = createModel(train_data, train_result)
    costView(BGD_train_cost, SGD_train_cost, MBGD_train_cost)
