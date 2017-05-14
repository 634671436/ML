#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/15  14:14
---------------------------
    Author       :  WangKun
    Filename     :  KMeans.py
    Description  :  KMeans 聚类算法的一种
                    无监督学习：数据没有附带任何标签，也即无监督学习的目标是找到数据的某种内在结构
                    聚类：聚类是一种无监督学习算法，它将相似的对象归到同一个簇中，它好像是全自动分类（但是这种分类连类别体系都
                是自动构建的)，聚类的方法可以应用于所以的对象，簇内对象越相似，聚类效果越好。包括但不限于K-means、二分K-means
                等。
                    K-means（K均值）：最普通的聚类算法。K值代表它可以发现k个不同的簇，且每个簇的中心采用簇中所含值的均值计算而
                成。K-means采用迭代算法，其伪代码如下：
                    1、首先选择K个随机的点，称为聚类中心
                    2、对于数据集中的每一个数据，计算它与K个中心点的‘距离’，将其与距离最近的中心点关联起来，与同一个中心点关
                联的所有点聚成一类。
                    3、计算每个簇的平均值，将该组所关联的中心点位置移动到平均点处
                    4、重复步骤2-4至中心点不在发生变化

                    针对以上伪代码，重点关注点：1、K值是如何选定的？2、最初始的K个随机点是如何选定的？3、'距离'如何度量？4、如
                何确定聚类的好坏，是否存在局部最优解，也即如何评判聚类好坏？
                    答：1）一般而言，没有自动的选择聚类数量K的方法，主要通过：人工经验  下游的分类目标（衣服的M/L/XL/XXL/XXXL）
                    结合SSE（误差平方和）的‘肘部法则’
                        2）最初始的K值一般是随机生成的，一定要‘随机’。
                        3）距离的度量从KNN就开始，一般包括欧氏距离，曼哈顿距离（城市街区距离），切比雪夫距离，但是欧氏距离对于
                    更为普遍
                        4）K-均值存在局部最优解的情况。
                           度量聚类效果的指标是SSE（Sum of Squared Error，误差平方和），也即NG课中的Cost Founction，SSE越小
                    表明数据点越接近于它们的质心，聚类效果也越好。
                            如何通过SSE指标去改善聚类效果：将SSE最大的簇包含的点过滤出来并对其进行二分聚类（再划分为两簇）。同
                    时为了保持总的簇数量不变，再将某两个簇进行合并（合并最近的质心或者合并两个使得SSE增幅最小的质心）。
---------------------------
"""

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(filename):
    dataSet = []
    with open(filename, 'r')as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            fLine = map(float, line)
            dataSet.append(fLine)
    return dataSet


def disEclud(vecA, vecB):
    '''
    ‘距离’度量：欧式距离
    :param vecA: 向量A
    :param vecB: 向量B
    :return: 向量之间的欧式距离
    '''
    return sqrt(sum(power(vecA-vecB, 2)))


def randCent(dataSet, K):
    '''
    生成k个随机的质心集合
    :param dataSet: 数据集
    :param K:
    :return:
    '''
    n = shape(dataSet)[1]
    centroid = mat(zeros((K, n)))
    for i in range(n):
        minJ = min(dataSet[:, i])         # eg: [[-5.37]]
        # minJS = float(minJ)             matrix --> float
        rangeJ = float(max(dataSet[:, i]) - minJ)
        centroid[:, i] = minJ + rangeJ * random.rand(K, 1)
    return centroid


def Kmeans(dataSet, k, disMeans=disEclud, createCent=randCent ):
    '''

    :param dataSet: 输入数据集，矩阵
    :param k: 聚类点数
    :param disMeans: 距离度量
    :param createCent: 创造聚类中心点
    :return:
    '''
    m = shape(dataSet)[0]
    n = shape(dataSet)[1]
    clusterAssment = mat(zeros((m, n)))   # 二维矩阵，第一维将赋值为每个点所属的簇，第二维赋值为误差
    centroid = createCent(dataSet, k)     # 随机创建k个质心

    clusterChange = True
    while clusterChange:
        clusterChange = False
        for i in range(m):              # 对每个数据
            minDis = inf
            minIndex = -1
            for j in range(k):             # 对每个质心
                Mindistance = disMeans(dataSet[i, :], centroid[j, :])
                if Mindistance < minDis:
                    minDis = Mindistance
                    minIndex = j
            #  clusterChange = True 在整体不同循环的时候发挥作用，即第一次和第二次
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
            clusterAssment[i, :] = minIndex, minDis**2
        # print centroid
        for cent in range(k):    # 更新质心的位置
            ptsInclust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroid[cent, :] = mean(ptsInclust, axis=0)
    return centroid, clusterAssment


if __name__ == '__main__':
    dataSet = loadDataSet('testSet.txt')
    dataMat = mat(dataSet)
    centroid = randCent(dataMat, 2)
    distance = disEclud(dataMat[0], dataMat[1])
    print distance

    myCentroid, clusterAssment = Kmeans(dataMat, 4)
    print myCentroid
    print clusterAssment

    locX = []
    locY = []
    for locat in myCentroid.getA().tolist():
        locX.append(locat[0])
        locY.append(locat[1])
    print locX
    print locY

    dataX = []
    dataY = []
    with open('testSet.txt', 'r')as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            dataX.append(line[0])
            dataY.append(line[1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataX, dataY)
    ax.scatter(locX, locY, color='red')
    plt.show()

