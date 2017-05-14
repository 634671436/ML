#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/14  10:07
---------------------------
    Author       :  WangKun
    Filename     :  modelTree.py
    Description  :  树回归
                    1、模型树：
                        将叶子节点设置为分段线性函数，这里的分段线性是指模型由多个线性片段组成
                    2、比较：
                        决策树相对于其他机器学习学习算法的优势就在于结果更容易理解
                        模型树的可解释性显然是要优于回归树，而且模型树的预测准确度也更高。
                    3、理解：
                        模型树与回归树的区别：模型树的每个单独的叶子节点都用Linear Regressi去拟合，误差计算用数据集平方误差计
                    算modelErr()。只是在叶子节点上的表示上出现了不同的方式。

---------------------------
"""
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(filename):
    dataMat = []
    with open(filename, 'r')as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            # 高阶内置函数，将每一行映射为浮点数, 对序列中的每一个元素都进行处理，for循环的简化版
            fltLine = map(float, line)
            dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    '''
    二元切分数据集
    :param dataSet: 待切分数据集
    :param feature: 待解分的特征
    :param value: 该特征的某个值
    :return: 切分好的数据集
    '''
    '''
    nonzero()函数以元组方式返回满足条件的坐标目录，所以这里的nonzero()[0]返回的是所有满足条件的行的坐标
    '''
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def linearSolve(dataSet):
    '''
    分段计算Linear Regression
    :param dataSet: 数据集
    :return: 拟合直线的权值
    '''
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:-1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular,connot to inverse,\ntry increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):
    '''
    生成叶子节点模型
    :param dataSet:数据集
    :return: 权值ws
    '''
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    '''
    计算模型误差
    :param dataSet: 数据集
    :return: 在给定数据集上计算误差
    '''
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y-yHat,2))


def createTree(dataSet, leafType=modelLeaf, errType=modelErr, ops=(1, 4)):
    '''
    树构造函数
    :param dataSet: 数据集
    :param leafType: regLeaf建立叶节点函数
    :param errType: regErr 代表误差计算函数
    :param ops:
    :return:
    '''
    '''
    chooseBestSplit()函数，如果构建的是回归树，则返回的模型为常数；如果构建的时模型树，返回的则是线性方程
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:  # 满足停止条件时，返回叶子节点值，这里也只递归的出口
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def chooseBestSplit(dataSet, leafType=modelLeaf, errType=modelErr, ops=(1, 4)):
    # 用户设定的参数，用于控制函数的停止时机，具体的tolS是容许的误差下降值；tolN是切分的最小样本树
    tolS = ops[0]
    tolN = ops[1]

    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 如果所有的值都相等，则退出
        return None, leafType(dataSet)

    m, n = shape(dataSet)
    S = errType(dataSet)     # 在整个数据集上用一条直线拟合，计算误差
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):  # 意即：选择原来样本中的每一维特征的每一种可能取值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 如果切分出来的数据集很小，就不应该进行切分
                continue
            newS = errType(mat0) + errType(mat1)   # 对每个数据集分别用直线拟合，计算误差
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:  # 如果切分数据集之后的方差改变不大，则直接生成叶子，用一条直线拟合
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue

if __name__ == '__main__':
    myMat2 = mat(loadDataSet('exp2.txt'))
    Tree = createTree(myMat2,ops=(1,10))
    print Tree


