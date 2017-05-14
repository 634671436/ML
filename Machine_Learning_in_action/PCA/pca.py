#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/18  19:30
---------------------------
    Author       :  WangKun
    Filename     :  pca.py
    Description  :  PCA（Principal Compotent Analysis），主成分分析
                    无监督学习：数据没有附带任何标签，也即无监督学习的目标是找到数据的某种内在结构
                    降维：将数据从高维空间降低到低维，降维可见的优势包括：1、使得数据更容易使用；2、降低很多算法的计算开
                销；3、去除噪声（下面结合PCA再解释）；4、使得结果易懂。常见的降维方法包括但不限于：
                        1、PCA（Principal Compotent Analysis），主成分分析。
                        2、因子分析
                        3、独立成分分析（Indepedent Component Analysis ,ICA）
                    主要集中对PCA的学习：
                        解释：PCA不是针对原有的数据删减，而是把原来的数据映射到新的特征空间。一般认为，舍弃掉一些特征是
                    必要的。一方面，舍弃这些特征使的样本的采样密度增大；另一方面、从信号处理方面而言，认为信号的方差较大，
                    而噪声的信号方差较小，因此我们在丢弃一些方差较小的特征（特征值）时，相应的减少了噪声，因此起到了去噪
                    的作用。
                        算法思想：相同的一组数据，给定不同的的基，数据有不同的表示，当选择的基的数量少于向量本身维数时，
                    就能起到降维的作用。但是我们得选择一组基使其能够最大程度的保留原有的信息。此时为了能最大程度的保留原
                    有的信息，我们希望投影之后的数据尽量的分散，而这种分散在数学上可以用方差去衡量。二维空间上，我们想要
                    找到在一维空间上方差最大的投影，。。由此如果在多维甚至高维空间上，我们找到了第一个投影方向，还需要选
                    择第二个投影方向，第三个甚至更多。。。
                        但同时我们又不能让不同的方向上存在线性相关，因为线性相关意味着这两维不是完全独立，也就存在重复表
                    示的信息。这种关系在数学上我们通过协方差衡量，协方差是衡量不同变量（而不是不同样本）之间的相关性。
                        而协方差矩阵在最大化方差和协方差为0时具有了统一性，因此通过协方差矩阵求其解特征值和特征向量，选
                    择前k个特征值对应的特征向量，对角元素从大到小依次排序，各主成分方差依次递减，而选择的k个特征值对应的
                    特征向量就是投影方向的方向向量。

                        因此有如下算法步骤：
                        1、初始数据集零均值化（初始样本每一维均值为0，因此投影到基上的样本点的均值依旧为0，在求解时带来计
                    算上的方便）；
                        2、计算协方差矩阵
                        3、求出协方差矩阵对应的特征值和对应的特征向量
                        4、按大小顺序排序取得前k个特征值对应的特征向量，组成（n * k）的基矩阵
                        5、Y (m * k) = X (m * n) * P（n * k），将样本从X的n维转化为Y的k维。

                        注意：降维技术可以运用在监督学习和非监督学习中。
---------------------------
"""

from numpy import *


def loadDataSet(filename,delim='\t'):
    dataSet = []
    with open(filename, 'r')as f:
        for line in f.readlines():
            line = line.strip().split(delim)
            fLine = map(float,line)
            dataSet.append(fLine)
    return mat(dataSet)


def pca(dataMat, topNfeat = 9999):

    meanVals = mean(dataMat, axis=0)
    meanRemove = dataMat - meanVals              # 对输入矩阵零均值化

    covMat = cov(meanRemove, rowvar=0)                # 求协方差矩阵
    eigVals, eigVects = linalg.eig(mat(covMat))       #  eigVals,<type:ndarry> /求协方差矩阵的特征值和特征向量

    # argsort()返回的是数组值从小到大的索引值
    eigValInd = argsort(eigVals)                      # 从小到大对N个值排序
    eigValInd = eigValInd[:-(topNfeat + 1):-1]
    redEigVects = eigVects[:, eigValInd]

    lowDDataMat = meanRemove * redEigVects             # 将数据转化到新空间降维
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt')
    lowDDataMat, reconMat = pca(dataMat, 1)
    print shape(lowDDataMat)








