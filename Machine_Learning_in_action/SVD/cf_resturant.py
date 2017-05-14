#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/20  20:00
---------------------------
    Author       :  WangKun
    Filename     :  cf_restaurant.py
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

                    餐馆菜肴推荐引擎：
                        推荐系统的工作流程：给定一个用户，系统会为此用户返回N个最好的推荐菜。
                        1、寻找用户没有评级的菜肴，即在用户-物品矩阵中的0值
                        2、在用户没有评级的所有物品中，对每个物品预定一个可能的评价分数，这也就是我们认为用户可能会对物
                    品的评分。（这也就是相似度计算的初衷）
                        3、对这些评分从高到低进行排序，返回前N个物品

---------------------------
"""
from numpy import *
import svd_cf


def recommand(dataSet, user, N=3, simMeas = svd_cf.cosSim):

    # 找出所有未评分的物品
    unratedItems = nonzero(dataSet[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you have rated all items'
    itemScore = []
    for item in unratedItems:
        estimatedScore = standEst(dataSet, user, simMeas, item)
        itemScore.append((item, estimatedScore))

    # 寻找前N个未评级的物品（topN 推荐）
    return sorted(itemScore, key=lambda jj: jj[1], reverse=True)[:N]


def standEst(dataSet, user, simMeas, item):
    '''
    :param dataSet:  用户物品矩阵
    :param user:     对具体用户的推荐
    :param simMeas:  相似性的度量
    :param item:     指定user未评分的物品
    :return:         对未评分物品的打分估计
    '''
    n = shape(dataSet)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for i in range(n):
        userRating = dataSet[user, i]
        if userRating == 0:
            continue
        # 这里找出的用户满足两个条件，1、用户对user未评价的物品需要有评分dataSet[:, item].A > 0，
        #                             2、而且user有评分物品用户也需要评分dataSet[:, i].A > 0。
        #                             同时满足以上两点。
        overLap = nonzero(logical_and(dataSet[:, item].A > 0, dataSet[:, i].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0.0
        else:
            similarity = simMeas(dataSet[overLap, item], dataSet[overLap, i])
            print 'the %d and %d similarity is: %f' % (item, i, similarity )
        simTotal += similarity
        ratSimTotal += similarity * userRating                # A为用户评价物品，B未评价，B的评分等于A的评分 * 相似度
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal


def svdEst(dataSet, user, simMeas, item):

    n = shape(dataSet)[0]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,sigma,V = linalg.svd(dataSet)             # SVD矩阵分解
    Sig4 = mat(eye(4) * sigma[:, 4])            # 构建对角矩阵
    xformedItems = dataSet.T * U[:, :4] * Sig4.I         # 将高维转化为低维，构建转化之后的物品

    for i in range(n):
        userRating = dataSet[user,i]
        if userRating == 0:
            continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[i,:].T)
        print 'the %d and %d similarity is %f' % (item, i, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal


def loadExdata():
    return [[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]]

if __name__ == '__main__':
    myDat = mat(loadExdata())
    recommandList = recommand(myDat, 2)
    print recommandList




