#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/1  13:49
---------------------------
    Author       :  WangKun
    Filename     :  logRegres.py
    Description  :  逻辑回归（logistic Regression）
                    1、对于给定的实例，计算P（Y=1|X）和P（Y=0|X）,并比较两个条件概率的大小，将实例分类到概率大的一边
                    2、h（x） = P（Y=1|X），对于给定的输入变量，根据选择的参数计算输出变量取值为1的可能性
                    3、模型参数估计的两种思考：
                        1、从极大似然估计的解释出发：对于给定的数据集，我们可以把它看做是事实，而我们要做的就是得到参数为了使
                    模型更加接近与事实，也就是使由数据集X得到对于Y的可能性最大，也即对于数据集中数据而言，P（D|X）最大，显然
                    对于每一组数据都是相互独立的，因此P（Y|X,theta）= P（y1|X1,theta）*P（y2|X2,theta）*...
                        由于对于logistic Regession而言，输出即为概率，也即：P（yi|Xi,theta） = h（x）^yi + (1-h(x))^(1-yi)
                    因此，P（Y|X,theta） = （h（x1）^y1 + (1-h(x1))^(1-y1)）*（h（x2）^y2 + (1-h(x2))^(1-y2)）*。。。
                        再取对数就是书上的形式了
                        2、根据李航《统计学习方法》1.3.2节，损失函数或者代价函数的定义，logistic Regression选取对数似然损失
                    函数，具体的推导见Andrew NG 的视频
                    4、在得到J（theta）后，我们需要最小化（最大化）代价函数时，用到的可以梯度下降（上升）法，如何判断何时迭代
                    停止，一是迭代次数达到某个固定次数；二是代价函数达到某个可以允许的误差范围
---------------------------
"""


'''
  值得注意的点：
     1、在绘图的时候，plotBestFit(weights.getA())中的weights.getA()
'''
from math import exp
from numpy import *
import matplotlib.pyplot as plt

def loadData():
    '''
    读取testSet.txt的内容
    :return: 训练矩阵（2维扩充为3维，每行第一列为1），标签矩阵
    '''
    dataMat = []
    labelMat = []
    with open('testSet.txt')as f:
        for line in f.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([1,float(lineArr[0]),float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


def sigmod(inX):
    '''
    :param inX:
    :return: sigmod函数结果
    '''
    return 1.0/(1+exp(-inX))

'''
    算法一（随机梯度上升算法）：
    1、采用批梯度算法的方式
    2、对迭代次数限制，即通过达到固定的次数停止迭代
    3、使用梯度上升的方式求取最值，因为theta：= theta + ... 是根据代价函数的求导得到了，而代价函数的形式是一定的，
   如果代价函数表示不同，则梯度方式就不同
'''
def gradAscent(dataMatIn, classLabels):
    '''
    :param dataMatIn: 训练举证
    :param classLabels: 标签矩阵
    :return: 权值
    '''
    dataMatrix = mat(dataMatIn)                # 转换为numpy矩阵数据类型
    labelMatrix = mat(classLabels).transpose()

    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))

    for k in range(maxCycles):
        h = sigmod(dataMatrix * weights)
        error = (labelMatrix - h)
        weights = weights + alpha * dataMatrix.transpose() * error      # 在numpy中 * 则是矩阵相乘
    return weights

'''
    算法二：随机梯度上升算法
    当样本的数量和特征的数量达到上千万时，批梯度算法的计算复杂度就太高了，因此可以利用每个样本来更新回归系数
    随机梯度上升算法，也即在线学习算法
    1、该算法与批梯度算法不同点在于：
        1、这里的h和error都是数值，而批梯度上升算法为向量
        2、这里的变换中，所有的数据类型都是数组

    但是这里的实际分割效果并不好，主要原因是因为：
        1、批梯度上升算法是在整个数据集上运行了500次，而随机梯度上升算法仅仅只是运行了500条单独的数据，因此对算法
    做一些修改


'''
def stocGradAsent0(dataMatrix,classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmod(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

'''
    随机梯度上升算法的修改：
    1、'随机':数据集中的数据是随机的挑选，不定顺序的运行；避免数据集周期性的波动影响。
    2、在整个数据集上运行多次
    3、alpha的值在每次迭代中变化，经过试验，改变alpha会使得收敛速度更快
'''
def stocGradAsent1(dataMatrix, classLabels,CycleNum):

    m,n = shape(dataMatrix)
    weights = ones(n)
    # alpha = 0.01
    for i in range(CycleNum):
        dataIndex = range(m)   # dataIndex <type:list>, [0,1,2,3,4,5,6,7,8...]
        for j in range(m):
            alpha = 4/(1.0+i+j) + 0.01                       # alpha每次迭代的时候都会调整大小
            randIndex = int(random.uniform(0, len(dataIndex)))         # '随机'更新，避免出现周期性的波动
            h = sigmod(sum(weights * dataMatrix[randIndex]))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(weights):
    '''
    绘制决策边界
    :param weights:   权重列表
    :return:  绘制好的图形
    '''
    dataMat, labelMat = loadData()
    dataArray = array(dataMat)
    n = dataArray.shape[0]
    x1cord = []; y1cord = []
    x2cord = []; y2cord = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            x1cord.append(dataArray[i,1])
            y1cord.append(dataArray[i,2])
        else:
            x2cord.append(dataArray[i,1])
            y2cord.append(dataArray[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1cord,y1cord,s=30, c='red', marker='s')
    ax.scatter(x2cord,y2cord,s=30, c='green')

    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]              # 最佳拟合曲线
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadData()
    print dataMat
    print labelMat

    # weights = gradAscent(dataMat, labelMat)
    # print weights               # 这里的weights为numpy.Matrix

    '''
        NumPy其实包含两种基本的数据类型：数组和矩阵
        在plotBestFit的y = (-weights[0]-weights[1]*x)/weights[2]中，此时y需要的是一个数组对象
    而通过weight.getA()就可以将其自身转化为数组对象
    '''
    # plotBestFit(weights.getA())

    # weights = stocGradAsent0(array(dataMat),labelMat)
    # plotBestFit(weights)

    weights = stocGradAsent1(array(dataMat), labelMat, 500)
    plotBestFit(weights)

