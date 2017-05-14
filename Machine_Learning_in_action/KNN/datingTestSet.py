#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/2/21  10:30
---------------------------
    Author       :  WangKun
    Filename     :  KNN-datingTestSet.py
    Description  :  k-nearest neighbor用于改进约会网站的配对效果

---------------------------
"""

from numpy import *
import matplotlib.pyplot as plt
import KNN


def file2Matrix(filename):      # 读取文本文件，将文件转化为矩阵
    with open(filename) as f:
        arrayOLines = f.readlines()       # arrayOLines的数据类型为list,包含所有行
        numberOLines = len(arrayOLines)
    returnMat = zeros((numberOLines, 3))
    classLabelVertor = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')    # lineFromLine的数据类型也是list
        returnMat[index, :] = listFromLine[0:3]
        classLabelVertor.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVertor

def autoNorm(dataSet):  # 归一化数据
    minVals = dataSet.min(0)           # ndarray，3列，把没列中最小的值放在变量minVals中
    maxVals = dataSet.max(0)           # ndarray，3列，把没列中最大的值放在变量maxVals中
    range = maxVals - minVals
    normalDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalDataSet = dataSet - tile(minVals, (m,1))
    normalDataSet = normalDataSet/tile(range, (m, 1))
    return normalDataSet, range, minVals

def datingClassTest(): # 测试分类器性能
    hoRatio = 0.1   # 抽取10%的测试数据
    returnMat, classLabelVertor = file2Matrix('datingTestSet2.txt')
    normalDataSet, ranges, minVals = autoNorm(returnMat)
    m = normalDataSet.shape[0]
    numTestVes = int(hoRatio*m)
    errorCount = 0

    for i in range(numTestVes):
        classfierResult = KNN.classify(normalDataSet[i,:],
                                       normalDataSet[numTestVes:m,:],classLabelVertor[numTestVes:m],4)
        print 'the classfier came back with: %d, the real answer id %d' % (classfierResult, classLabelVertor[i])
        if (classfierResult != classLabelVertor[i]): errorCount += 1
    print 'the total error rate is :%f' % (errorCount/float(numTestVes))

def classfyPerson():
    resultLists = ['not at all', 'in small doses', 'in large doses']
    PersontTats = float(raw_input('persentage of time spent playing video games?'))
    ffMiles = float(raw_input('frequent flier miles earned per year?'))
    icecream = float(raw_input('liters of ice cream consumed per year?'))

    datingDataMat, datingLabels = file2Matrix('datingTestSet2.txt')
    normalMat, ranges, minVals = autoNorm(datingDataMat)           # 这里保留的三个变量中范围和最小值就能归一化 测试值

    inArr = array([ffMiles,PersontTats, icecream])
    classifierResult = KNN.classify((inArr-minVals)/ranges, normalMat, datingLabels, 4)

    # print classifierResult
    print 'You will probably like this person:', resultLists[classifierResult-1]


if __name__ == '__main__':
    # returnMat, classLabelVertor = file2Matrix('datingTestSet2.txt')
    '''
        查看数据
    '''
    # print returnMat
    # print classLabelVertor
    '''
        绘图可视化数据
    '''
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(returnMat[:,1],returnMat[:,2],15.0*array(classLabelVertor),15.0*array(classLabelVertor))
    # plt.show()
    '''
        归一化数据
    '''
    # normalDataSet, range, minVals = autoNorm(returnMat)
    # print normalDataSet
    # print range
    # print minVals
    '''
        测试代码,当k为4时误差最小为4%
    '''
    # datingClassTest()

    classfyPerson()
