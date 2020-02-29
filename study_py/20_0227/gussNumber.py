#!/usr/bin/env python
# encoding: utf-8
'''
@author: wty
@license: (C) Copyright 2019-2020, Node Supply Chain Manager Corporation Limited.
@contact: wty229027377@gmail.com
@software: pycharm
@file: gussNumber.py
@time: 2020/2/27 23:51
@desc:
'''

import random


def secret():
    secret_num = random.randint(1, 20)
    return secret_num


def guesses(secret_num):
    for a in range(1, 20):
        print("please guesses this secret num")
        guess = int(input())
        if guess < secret_num:
            print("low")
        elif guess > secret_num:
            print("high")
        else:
            print("success")
            break


if __name__ == '__main__':
    num = secret()
    guesses(num)
