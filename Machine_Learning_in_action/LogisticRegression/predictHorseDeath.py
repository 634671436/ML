#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/2  10:42
---------------------------
    Author       :  WangKun
    Filename     :  PredictHorseDeath.py
    Description  :  逻辑回归（logistic Regression）
                    从疝气病预测病马的死亡率，使用Logistic回归来预测患有疝气病的马的死亡率
---------------------------
"""
'''
    值得注意的点：
    1、由于在准备数据的过程中，发现30%的数据是缺失的，因此需要对缺失值进行处理，处理办法如下：
        1、使用可用特征的均值来填补缺失值
        2、使用特殊值来填补缺失值，如-1等
        3、忽略有缺失值的样本
        4、使用相似样本的均值填补缺失值
        5、使用另外的机器学习算法预测缺失值
'''
from logRegres import *


def classifyVector(inX, weights):
    prob = sigmod(sum(inX*weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def colicTest():

    TrainingSet = []
    ClassLabel = []

    with open('horseColicTraining.txt')as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(line[i]))
            lineArr.insert(0,1.0)
            TrainingSet.append(lineArr)
            ClassLabel.append(float(line[21]))             # 组建好训练数据集

    trainingWeights = stocGradAsent1(array(TrainingSet), ClassLabel, 500)

    errorCount = 0.0
    numTestVec = 0.0
    with open('horseColicTest.txt')as f:
        for line in f.readlines():
            numTestVec += 1
            line = line.strip().split('\t')
            lineArr  = []
            for i in range(21):
                lineArr.append(float(line[i]))
            lineArr.insert(0, 1.0)
            if int(classifyVector(array(lineArr), trainingWeights)) != int(line[21]):
                errorCount += 1
    errorCountRate = float(errorCount)/numTestVec

    print 'the error rate of this test is: %f' % errorCountRate
    return errorCountRate

def multiTest():
    numTest = 10;
    errorSum = 0.0
    for k in range(numTest):
        errorSum += colicTest()
    print 'after %d iterations the average error rate is :%f' %(numTest,errorSum/float(numTest))

if __name__ == '__main__':
    # TrainingSet, ClassLabel = colicTest()
    # print TrainingSet
    # print ClassLabel

    # colicTest()
    multiTest()

