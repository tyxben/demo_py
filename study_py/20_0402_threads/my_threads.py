#!/usr/bin/env python
# encoding: utf-8
'''
@author: wty
@license: (C) Copyright 2019-2020, Node Supply Chain Manager Corporation Limited.
@contact: wty229027377@gmail.com
@software: pycharm
@file: my_threads.py
@time: 2020/4/2 23:21
@desc:
'''

import threading
import time


def run(n):
    print("task", n)
    time.sleep(1)
    print('2s')
    time.sleep(1)
    print('1s')
    time.sleep(1)
    print('0s')
    time.sleep(1)


def cal_sum(begin, end):
    # global _sum
    _sum = 0
    for i in range(begin, end + 1):
        _sum += i
    return _sum


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


if __name__ == '__main__':
    t1 = MyThread(cal_sum, (1, 2))
    t2 = MyThread(cal_sum, (2, 3))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1 = t1.get_result()
    res2 = t2.get_result()
    print(res1 + res2)
