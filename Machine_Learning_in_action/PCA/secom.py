#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/18  20:30
---------------------------
    Author       :  WangKun
    Filename     :  secom.py
    Description  :  利用主成分分析方法降维
                    数据来源为半导体制造数据，590个特征
                    数据处理：
                        1、用平均值补全缺失值
                        2、PCA降维
---------------------------
"""

import pca
from numpy import *


def replaceNanWithMean():
    datMat = pca.loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]

    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])       # 求非NAN值的平均值

        datMat[nonzero(isnan(datMat[:, i]))[0], i] = meanVal

    return datMat

if __name__ == '__main__':
    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals

    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))

    print eigVals






