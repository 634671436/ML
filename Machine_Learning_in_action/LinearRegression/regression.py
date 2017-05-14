#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/8  9:52
---------------------------
    Author       :  WangKun
    Filename     :  regression.py
    Description  :  线性回归
                    1、线性回归的损失函数是平方损失函数，L(Y,f(X)) = (Y-f(X))^2

                    局部加权线性回归(LWLR)：
                    1、在线性回归发生欠拟合的时候，在估计中引入一些偏差，降低预测的均方误差。
                    2、我们给待预测的点附近的点一定的权重，而使得远离它的点权重较低
                    3、非参数学习方法：
                        1、有参数学习方法是啥？eg：LR。在训练完所有数据之后得到一系列训练参数，然后根据训练参数来预测样本的值，
                    这时不再依赖之前的训练数据，参数是确定的
                        2、非参数学习方法：eg：LWLR。在预测新样本值时每次都会重新训练新的参数，也就是每次预测新的样本值都会依
                    赖训练数据集合，所以每次的参数是不确定的。
                    4、局部加权线性回归的缺点：
                        对每个点做预测时都必须使用整个数据集，因此当训练容量过大时，非参数学习算法需要占用更多的存储容量，计算
                    速度较慢
---------------------------
"""

'''
    个人的理解：
        1、局部加权线性回归，对于样本的预测，其实就是对每个点的预测（或者说拟合），样本点有m个点，则预测出来的也是m个点
        ，而这m个点不在一条直线上，而是离散的点，因此最终拟合的'线性'（实为非线性）曲线则是m个点的（m-1条线段）组成的估
        计曲线。

        2、在对测试集上的样本点进行预测时，需要依赖于训练集数据集合，因为它需要计算该点与所有点的'距离'，来给出不同的
        权值。

        3、而对训练集进行拟合时，也是同上述相同的过程，重复了m遍（假设m个样本），得到m个结果。
            其实也没必要进行拟合，因为拟合出来的参数都没意义，每次都变化了。

        4、简单回顾一下局部加权线性回归来拟合一下模型过程：

            1.用高斯核函数计算出第ｉ个样本处，其它所有样本点的权重Ｗ
            2.用权重ｗ对第ｉ个样本作加权线性回归，得到回归方程，即拟合的直线方程
            3.用刚才得到的经验回归直线计算出xi处的估计值y^i
            4.重复一至三步，得到每个样本点的估计值
'''
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(filename):
    '''
    加载数据
    :param filename:文件名
    :return:
    '''
    numFeat = len(open(filename).readline().split('\t')) - 1

    dataMat = []
    Labels = []

    with open(filename,'r')as f:
        for line in f.readlines():
            lineArr = []
            line = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(line[i]))
            dataMat.append(lineArr)
            Labels.append(float(line[-1]))
    return dataMat, Labels


def plotData(filename):
    '''
    描点,去第二列、第三列描点
    :param filename: 文件名
    :return:
    '''
    data = []
    Label = []
    with open(filename,'r')as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            data.append(line[1])
            Label.append(line[2])
    '''
      描点
    '''
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.scatter(data, Label, c='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return data,Label

'''
    LR,线性回归
'''
def standRegres(xArr, yArr):
    '''
    利用矩阵直接计算权值w
    :param xArr: 训练矩阵
    :param yArr: 训练数据标签
    :return:  权值w
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T

    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print 'This matrix is singular,cannot do inverse'
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

'''
    LWLR，局部加权线性回归
    lwlr（）测试每个点；lwlrTest（）测试数据集
'''
def lwlr(testPoint,xArr,yArr,k=1.0):
    '''
    给定x空间中一点，计算出对应的预测值yHat。
    :param testPoint: 待测点，也即每次预测的新的样本值
    :param xArr:
    :param yArr:
    :param k:
    :return:
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = xMat.shape[0]
    weight = mat(eye((m)))

    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weight[j,j] = exp(diffMat * diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weight * xMat)
    if linalg.det(xTx) == 0.0:
        print 'This matrix is singular, cannot do inverse'
        return
    ws = xTx.I * (xMat.T * (weight * yMat))
    return testPoint * ws


def lwlrTest(testArr,xArr,yArr,k=1.0):
    '''
    每次预测新的样本值都会依赖训练数据的集合，这里可以看到对于每个数据，都会调用前面的函数，而前面的函数又依赖于整个数据集
    :param testArr: 测试数据集
    :param xArr:训练数据集的样本
    :param yArr:训练数据集的标签
    :param k:衰减系数
    :return:
    '''
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def lwlrplot(yHat, xArr, yArr):

    xMat = mat(xArr)
    strInd = xMat[:,1].argsort(0)
    xSort = xMat[strInd][:,0,:]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(xSort[:,1], yHat[strInd])
    # ax.scatter(xMat[:,1].flatten().A[0], yHat, s=2, c='blue')
    ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0], s=2, c='blue')
    plt.show()

if __name__ == '__main__':
    dataMat, Labels = loadDataSet('ex0.txt')
    print dataMat       # tpye<list>
    print Labels

    # data,Label = plotData('ex0.txt')
    # print data
    # print Label
    '''
    1、LR部分
    '''
    # ws = standRegres(dataMat, Labels)
    # print ws
    #
    # xMat = mat(dataMat)    # type<numpy.matrix>
    # yMat = mat(Labels)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    '''
        描点是利用numpy中的矩阵选择x的第二列，和y描点
            xMat[:,1].flatten().A[0]中flatten()函数将二维矩阵转化成一维矩阵，
        mtrix.A[0]和getA()意思相同，将矩阵转化为一维数组
    '''
    # ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
    #
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat = xCopy * ws
    # ax.plot(xCopy[:,1], yHat)
    # plt.show()
    '''
    2、LWLR
    '''
    yHat = lwlrTest(dataMat,dataMat,Labels,0.003)
    print yHat
    # print type(yHat)   # <type 'numpy.ndarray'>
    lwlrplot(yHat,dataMat,Labels)









