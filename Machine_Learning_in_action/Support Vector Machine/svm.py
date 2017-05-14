#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/3  11:16
---------------------------
    Author       :  WangKun
    Filename     :  svm.py
    Description  :  支持向量机
                    1、定义：
                        支持向量机模型定义在特征空间上的间隔最大的线性分类器，其学习策略是间隔最大化，最终将问题转化为
                    一个凸二次规划问题的求解
                    2、分类：
                        1、线性可分支持向量机（数据集线性可分）
                        2、线性支持向量机（数据集近似可分）
                        3、非线性支持向量机（线性不可分）
                    3、线性
                        线性问题的转化最开始是通过最大化间隔，也即
                                    最大化几何间隔，满足约束条件超平面上任意训练样本的几何间隔至少为r
                                    最终转化为：
                                              min 1/2*||w||^2
                                        s.t.  y(w^T*x + b) >= 1
                                    而这也是一个凸二次规划问题

                        凸二次规划问题通过拉格朗日对偶性将其转化为对偶变量的最优化问题
                    4、SMO算法：
                        在问题转化过程中引入变量alpha，并最终转化为关于alpha的问题，SMO算法的目的是求出一系列alpha，
                    就能得出w和b，确定超平面。
                        原理：每次循环中选择两个alpha进行优化处理，一旦找到一对合适（满足一定条件）的alpha，那么久增大
                    其中一个同时减小另一个。这里所谓的‘合适’是指两个alpha必须要符合一定的条件，条件之一就是这两个alpha
                    必须要在间隔边界之外，而其二则是这两个alpha还没进行区间化处理或者不在边界上。
---------------------------
"""

def loadData(filename):
    dataMat = []
    labelMat = []
    with open(filename,'r')as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            dataMat.append([float(line[0]),float(line[1])])
            labelMat.append(float(line[2]))
    return dataMat,labelMat



