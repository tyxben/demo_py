#!/usr/bin/env python
# encoding: utf-8
'''
@author: wty
@license: (C) Copyright 2019-2020, Node Supply Chain Manager Corporation Limited.
@contact: wty229027377@gmail.com
@software: pycharm
@file: filesIO.py
@time: 2020/3/1 20:41
@desc:
'''

import os

if __name__ == '__main__':
    print(os.path.join('usr', 'bin', 'spam'))
    print(os.path.abspath('.'))
    x = y = z = 1
    print(x, y, z)
    print(x, y, z)
    x += y
    x = y = z + 1
    gen = (item for sub_list in [[1, 3, 5], [2, 4, 6], [3, 6, 9]] if 3 in sub_list for item in sub_list if
           item >= 3)
    for i in gen:
        print(i)
    print(1.2 - 1.0 == 0.2)
    a = [i for i in map(lambda x, y: x + y, ['hui', 'kaike', 'Artificial '], ['ke', 'ba', 'Intelligence'])]
    print(a)
    c = (1)
    b = (1,)
    a = (1, 2, (3, 4))
    stra = "DASDASDASD dsad ".split()
    for i, item in enumerate(stra):
        print(i, item)
    s = {'1': 1, '2': 2}
    theCopy = s.copy()
    s['1'] = 5
    sum = s['1'] + theCopy['1']
    print(sum)
