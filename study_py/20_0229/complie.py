#!/usr/bin/env python
# encoding: utf-8
'''
@author: wty
@license: (C) Copyright 2019-2020, Node Supply Chain Manager Corporation Limited.
@contact: wty229027377@gmail.com
@software: pycharm
@file: complie.py
@time: 2020/3/1 17:30
@desc:
'''

import re
import pyperclip


# 函数前带 "_" 表示 只能包内调用
def _specialWord():
    wordRegex = re.compile(r'[aeiouAEIOU]')
    k = wordRegex.findall('RoboCop eats baby food. BABY FOOD.')
    print(k)


# 处理 ".*" 匹配字符串
def _regex():
    namaRegex = re.compile(r'First Name: (.*) Last Name: (.*)')
    mo = namaRegex.search('First Name: AL Last Name: Sweigart')
    print(mo.group(1))
    print(mo.group(2))


def _subWord():
    subRegex = re.compile(r'Agent \w+')
    test = subRegex.sub('CENSORERD', 'Agent Alice gave the secret documents to Agent Bob.')
    print(test)

    subComile = re.compile(r'Agent (\w)\w*')
    test2 = subComile.sub(r'\1****', 'Agent Alice told Agent Carol that Agent Eve knew Agent Bob was a double agent.')
    print(test2)

    return


def _regPhone():
    """
    方法用于: 匹配email
    """
    phoneRegex = re.compile(r'''
       (\d{3}|(\d{3}\))?
       (\s|-|\.)?
       (\d{3})
       (\s|-|\.)
       (\s*(ext|x|ext.)\s*(\d{2,5}))?
       )
    ''', re.VERBOSE)

    return phoneRegex


def _regEamil():
    """
    方法用于: 匹配 email
    """
    emailRegex = re.compile(r'''[a-zA-Z0-9._%+-]+
    @
    [a-zA-Z0-9.-]+
    (\.[a-zA-Z]{2,4})
    ''', re.VERBOSE)

    return emailRegex


if __name__ == '__main__':
    # _specialWord()
    #
    # _regex()
    #
    # _subWord()

    phone = _regPhone()
    email = _regEamil()
    text = str(pyperclip.paste)
    matches = []
    for groups in phone.findall(text):
        phoneNum = '-'.join([groups[1], groups[3], groups[5]])
        print(len(groups))
        if groups[7] != '':
            phoneNum += ' x' + groups[8]

    for groups in email.findall(text):
        matches.append(groups[0])

    if len(matches) > 0:
        pyperclip.copy('\n'.join(matches))
        print('Copied to clipboard:')
        print('\n'.join(matches))
    else:
        print('No phoneNumber or email address found.')
