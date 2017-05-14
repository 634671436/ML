#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/6  10:42
---------------------------
    Author       :  WangKun
    Filename     :  Adaboosting_decisionStump.py
    Description  :  单层决策树构建弱分类器
                    Adaboosting（Adaptive Boosting，即自适应增强）：
                    1、“三个臭皮匠顶个诸葛亮”
                        对于分类问题，给定训练样本集，求比较粗糙的分类规则较为容易。Adaboosting算法即是通过从弱学习算法
                    出发，反复学习，得到一系列的弱分类器，然后组合这些分类器。
                    2、Adaboosting算法通过如下两个方法来自适应增强：
                        1、在每一轮训练中改变训练样本的权值，提高前一轮弱分类错误分类的样本的权值，加大数据权值有利于在后
                    面的弱分类器中被重点关注
                        2、加大分类错误率较低的弱分类器的权值（alpha），使其在表决中能其较大的作用
                    3、Adaboosting提供的是一个算法框架
                        在Adaboosting中可以选择不同的弱分类器（即：弱分类算法）进行组合，这些算法既可以是同一种算法，也可
                    以是不同的算法
                        这里采用的就是单层决策树构建的弱分类器，基于单个特征来做决策
---------------------------
"""

'''
    使用AdaBoosting算法值得注意的点：
    1、弱分类器D1、弱分类器D2的递进关系：
        弱分类器之间相互交互的地方发生在样本权值weightedError，不同的分类器在做分类之前可能就是完全相同的单层决策
    树模型，但是由于在每个弱分类器所有可选的单层决策树中需要选择weightedError最小的，而weightedError是不断的改变
    的，因此也使得下一个弱分类器直接发生了变化。
    2、对于单层决策树的理解：
        首先理解最低错误率单层决策树含义：也即只在某一维特征（单层）之上，构建决策树（标称性数据上使用'lt'、'gt'）
    决策树（标称型数据集）上的判断结点，可以设定为任何的离散点。这里需要保证的是错误率最低，也即加权数据集上的错误
    率。
    3、代码写法上：
        1、点乘multiply，类似于matlab中的点乘
        2、
'''
from numpy import *


def loadSimpleData():
    '''
    创造简单的数据集
    :return:
    '''
    dataMat = matrix([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat,classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    用于在dataMatrix的某一维上与某个阈值，做不等比较
    :param dataMatrix: 训练数据集
    :param dimen: 索引的index，从0开始
    :param threshVal:  在某一维度（索引）上的阈值
    :param threshIneq:  ['lt','gt'] => ['<','>']
    :return:
    '''
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':          # 下面赋值为-1，是因为errArr = mat(ones((m,1)))，值不同时需要改变值，但是没有-1的Array
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0     # 列表中的每一项都与threshVal比较
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray


'''
    构建最低错误率的单层决策树
    将最小错误率minError设为无穷大，也即 minError = numpy.inf
    对数据集中的每一个特征（第一层循环）
        对每一个步长（第二层循环）
            对每个不等式（第三层循环）
                建立一个单层决策树并利用加权数据集对其进行测试
                如果错误率低于minError时，则将当前单层决策树设为最佳单层决策树

    个人理解：
        首先理解最低错误率单层决策树含义：也即只在某一维特征（单层）之上，构建决策树（标称性数据上使用'lt'、'gt'）
    决策树（标称型数据集）上的判断结点，可以设定为任何的离散点。这里需要保证的是错误率最低，也即加权数据集上的错误
    率。
               x2   |
              max———————————
        >    ___|___|_______________|___
       lt_______|___|_______________|___________     |  stepSize = (max-min)/numSteps
             ___|___|_______________|___
        <=   ___|___|_______________|___
             ___|___|_______________|
             ___|___|_______________|
             ___|___|_______________|
              min———————————x1(max)
                x1  |
'''

def buildStump(dataArray, classLabels, D):
    '''
    弱分类器，构建最好的分类决策树桩
    :param dataArray: 待分类的数据集
    :param classLabels: 带标签的分类训练数据集标签
    :param D:  每个样本的权值
    :return:
        bestClassEst：类别估计
    '''

    dataMatrix = mat(dataArray)
    LabelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    minError = inf                  # 无穷大
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps      # stepSize = 0.1 也即步长 threshVal = rangeMin + 0.1 * j
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == LabelMat] = 0          # 两个列表中的每一项都进行比较，如果相等就是0,留下不等的1
                weightedError = D.T * errArr      # 带权值的error，由于在每一次的循环中权值变化，故最优的分类器也会发生变化
                # print 'split: dim %d, thresh %.2f, thresh inequal: %s, the weighted ' \
                #       'error is %.3f' %(i, threshVal, inequal, weightedError)

                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['inequal'] = inequal
    return bestStump,minError,bestClassEst


def adaBoostTrainDS(dataArr,classLabels,NumIt=40):
    '''
    利用单层决策树的弱分类器，来组件强可学习分类器，可以看到每次的弱分类器训练都是从0开始，只是权值发生了变化
    :param dataArr:  训练矩阵
    :param classLabels:  类别标签向量
    :param NumIt:    弱分类器的循环次数，在Adaboosting中唯一需要人为指定的参数
    :return:   总的分类结果
    '''
    weakClassArr = []
    m = dataArr.shape[0]
    D = mat(ones((m,1))/m)    # 初始化权重

    aggClassEst = mat(zeros((m,1)))
    for i in range(NumIt):             # 弱分类器的个数指定
        bestStump, error, classEst = buildStump(dataArr,classLabels,D)
        # print 'D:', D.T

        # 这里的max(error,1e-16)用于确保在没有错误的时候不会发生除零溢出
        alpha = float(0.5 * log((1 - error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print 'classEst:', classEst.T

        # 矩阵相乘，类似于matlab中的点乘，算法中的w(m+1) = w(m) * exp(-alpha * classLabel * G(x(i)))/Z(m)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()                         # 这里保证权值总和为1

        # 这里的aggClassEst是没有使用sign时，对每个样本的预测
        #                         |  -1   ; x < 0
        #           sign（x） =   |
        #                         |   1   ; x > 0
        aggClassEst += alpha * classEst
        # print 'aggClassEst: ', aggClassEst.T

        # 每一次最终的预测，对 sign(aggClassEst)!= mat(classLabels).T, ones((m,1)) 判断，如果不等则为1，即分类错误，等则0
        # 这里对分类正确错误一个较好的写法
        aggErrors = multiply(sign(aggClassEst)!= mat(classLabels).T, ones((m,1)))

        errorRate = aggErrors.sum()/m

        print 'the %d ,totla error: %f' % (i,errorRate),'\n'
        if errorRate == 0:
            break
    return weakClassArr


def adaClassify(datToClass, classifierArr):
    '''
      抽象出来的只给出分类结果的函数
    :param datToClass:  待分类的样本
    :param classifierArr:  分类器的集合 [{'dim':0,'ineq':'lt','thresh':1.3, 'alpha':0.896723},{'dim':1,'ineq':'gt',...}]
    :return:
    '''
    dataMatrix = mat(datToClass)
    m = dataMatrix.shape[0]
    aggClassEst = mat(zeros((m,1)))

    for i in range(len(classifierArr)):
        # classEst 是估计的分类，eg：[1,-1,-1,1,-1]
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['inequal'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print aggClassEst
    return sign(aggClassEst)


if __name__ == '__main__':
    dataMat, classLabels = loadSimpleData()
    print dataMat,

    # retArray = stumpClassify(dataMat,0,1,'lt')
    # print retArray         # [[-1.],[1.],[1.],[-1.][1.]]

    # D = mat(ones((5,1))/5)
    # bestStump, minError, bestClassEst = buildStump(dataMat, classLabels, D)
    # print bestStump
    # print minError
    # print bestClassEst
    weakClassArr = adaBoostTrainDS(dataMat, classLabels, 9)
    print weakClassArr