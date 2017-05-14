#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/2/24  9:42
---------------------------
    Author       :  WangKun
    Filename     :  pickle1.py
    Description  :  决策树分类器的硬盘存储
                    1、需要使用到pickle模块序列化对象
---------------------------
"""
import pickle


def storeTree(inputTree,filename):
    with open(filename,'w') as f:
        pickle.dump(inputTree, f)

def grabTree(filename):
    with open(filename)as f:
        return pickle.load(f)

if __name__ == '__main__':
    myTree ={'surfing':{0:'no',1:{'flipper':{0:'no',1:'yes'}}}}
    storeTree(myTree, 'classfierStorage.txt')
    tree = grabTree('classfierStorage.txt')
    print tree