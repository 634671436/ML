#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/20  17:32
---------------------------
    Author       :  WangKun
    Filename     :  svd_cf.py
    Description  :  推荐系统的相似度度量
                        协同过滤（collaborative filtering）：集体智慧编程的一种，通过将用户和其他用户的数据进行对比实现
                    推荐。不在依赖于传统的属性来描述物品，而是利用用户对它们的评价来计算相似度。相似度度的度量：
                    1、欧氏距离：
                        相似度 = 1/（1 + 距离），将相似度约束到[0,1]之间。当距离为0时，相似度为1，当距离很大时，相似度趋
                    于0。
                    2、皮尔逊相关系数
                        皮尔逊相关系数相对于距离度量的优势在于，它对用户评级的数量级不敏感，某个人对于物品的评价全是5分，
                    另一人全是1分，皮尔逊相关系数度量的都是一样的。皮尔逊相关系数在[-1,1]之间。
                    3、余弦相似度
                        余弦相似度计算的是两个向量的夹角。如果夹角是90度，则相似度为0，如果夹角是0度，则相似度是1，取值
                    范围也在[-1,1]之间。
                                cos theta = AB/（||A|| ||B||）
                        ||A||为A的2向量范数。
---------------------------
"""

from numpy import *

# 计算余弦相似度
def euclidSim(inA, inB):

    return 1.0/(1.0 + linalg.norm(inA - inB))


# 计算皮尔逊相似度
def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


# 计算余弦相似度
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5 * (num/denom)

def loadExdata():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


if __name__ == '__main__':

    myDat = mat(loadExdata())

    print myDat[:, 0]  # 列向量
    euclid = euclidSim(myDat[:, 0], myDat[:, 4])
    euclid1 = euclidSim(myDat[:, 0], myDat[:, 0])

    pears = pearsSim(myDat[:, 0], myDat[:, 4])
    pears1 = pearsSim(myDat[:, 0], myDat[:, 0])

    cos = cosSim(myDat[:, 0], myDat[:, 4])
    cos1 = cosSim(myDat[:, 0], myDat[:, 0])

    print euclid, euclid1
    print pears, pears1
    print cos, cos1


