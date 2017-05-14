#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/2/20  19:56
---------------------------
    Author       :  WangKun
    Filename     :  KNN.py
    Description  :  k-nearest neighbor
                    1、给定训练数据集，对于新的数据输入，在训练集中找到与该实例最近的k个实例，
                这k个实例的多数属于某个类，则把输入实例分为该个类
                    2、 k值的选择   距离的度量   分类决策的规则
                    3、 伪代码的执行过程：
                        1、计算已知类别数据集中的点与当前点的距离；
                        2、按照距离递增次序排序；
                        3、选取与当前距离最小的k个点；
                        4、确定当前k个点所在类别的出现频率；
                        5、返回前k个点出现频率最高的类别作为当前点的预测分类
---------------------------
"""

from numpy import *
import operator


def creatDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    label = ['A', 'A', 'B', 'B']
    return group, label


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # Matrix = tile(inX,(dataSetSize,1))
    diffMat = tile(inX,(dataSetSize,1)) - dataSet    # numpy的tile函数重复某个数组，是把[0，0] 扩充为[[0, 0],[0, 0],[0, 0 ],[0, 0]],再相减
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances**0.5
    sortedDistIndicies = distance.argsort()   # 将distance(ndarray)的数据从小到大的顺序排列，提取其对应的index值返回
                                              #  这里提取索引值的好处在于能够与label中的标签对应起来
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 这里的dict的get方法用于返回指定键的值，如果不存在就为默认值0，就像是if key in : ...
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
    '''
     sorted 可以对所有的可迭代对象进行排序，并且返回一个新的可迭代对象
     函数使用如下：
        第一个参数为可迭代对象，这里的classCount分解为元组列表
        第二个参数为取待排元素的第几项进行排序，这里是取classCount={'B':2，'A':1}中的2和1进行排序
        第三个参数为为升序还是降序排列，True为降序，默认为升序
    '''

    return sortedClassCount[0][0]

if __name__ == '__main__':
    group, label = creatDataSet()
    print classify([0,0], group, label, 3)











