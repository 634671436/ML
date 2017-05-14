#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/6  17:27
---------------------------
    Author       :  WangKun
    Filename     :  predictHorseill.py
    Description  :  单层决策树构建弱分类器
                    预测马疝病的发生，训练的算法利用Adaboosting_decisionStump.py中训练的一系列弱分类器作为基础
---------------------------
"""
from numpy import *
import Adaboosting_decisionStump


def loadDataSet(filename):
    '''
    读文件
    :param filename: 读取的文件名
    :return:
    '''
    dataMat = []
    labelMat = []
    numFeat = len(open(filename).readline().split('\t')) - 1       # 这里的特征数量应该在打开文件之前检测

    with open(filename, 'r')as f:
        for line in f.readlines():
            lineArr = []
            line = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(line[i]))
            labelMat.append(float(line[-1]))
            dataMat.append(lineArr)

    return dataMat, labelMat

if __name__ == '__main__':
    trainingdatArr, traininglabelArr = loadDataSet('horseColicTraining2.txt')
    # print dataMat
    # print labelMat

    # classifierArray  是组合起来的分类器综合
    classifierArray = Adaboosting_decisionStump.adaBoostTrainDS(trainingdatArr, traininglabelArr, 50)

    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction = Adaboosting_decisionStump.adaClassify(testArr, classifierArray)

    error = mat(ones((67,1)))
    errorCount = error[prediction != mat(testLabelArr).T].sum()
    print float(errorCount/67)