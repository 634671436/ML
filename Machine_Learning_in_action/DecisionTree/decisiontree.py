#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/2/24  9:42
---------------------------
    Author       :  WangKun
    Filename     :  decision.py
    Description  :  决策树
                    1、 递归的选择最优的特征，并根据该特征对训练数据进行分割，使得对各个子数据
                    有一个最好的分类。这一过程对于着特征空间的划分，也对应决策树的构建
                    2、 特征选择   决策树的生成   决策树的修剪
                    3、 特征选择的过程：
                        ID3：信息增益   g(D,A) = H(D)-H(D|A)   信息增益最高的特征就是最好的选择
                        C4.5：信息增益比
                    4、决策树的生成：
                            递归生成决策树  结束条件：程序遍历完所有可能的划分数据集的属性，
                        或者每个分支下面的所有实例都属于同一个类，则终止循环
                            针对第一种如果遍历完所有的属性还没划分完，一般采用多数表决的方式划分
                    5、伪代码：
                       1、计算整个数据集的熵
                       2、划分数据集，计算熵
                       3、计算信息增益，选择最好的特征
                       3、利用选取的特征，递归构建决策树
---------------------------
"""
'''
   说明：
        数据要求：1、数据必须是列表元素组成的列表，而且所有的列表都要具有相同的数据长度；
                 2、数据的最后一列或者每个实例的最后一个元素是当前实例的标签
                 3、list中的数据类型无需限定
                 4、算法本身是不需要特征标签列表，但是决策树为了更可视化数据，都给出明确的含义
'''

from math import log
import operator


def createDataSet():
    dataSet = [[1,1,'yes'],                             # 创建数据集
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']             # 索引标签
    return dataSet,labels

def calcShannonEnt(dataSet):
    '''
    :param dataSet: 待计算熵的数据集
    :return: 该数据集的熵
    '''
    numEntries = len(dataSet)             # 可以用len计算numEntries，因为dataSet此时是2维list
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]

        '''
        这里对于字典的键是否存在的两种写法
        '''
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1

        # if currentLabel not in labelCounts.keys():
        #     labelCounts[currentLabel] = 0
        # labelCounts[currentLabel] += 1                    # 为dataSet的分类创建字典

    shannoEnt = 0.0
    for key in labelCounts:                  # labelCounts = {'yes':2, 'no':3}
        prob = float(labelCounts[key])/numEntries
        shannoEnt -= prob * log(prob,2)
    return shannoEnt


def splitDataSet(dataSet,axis,value):
    '''
    划分数据集，划分完的数据集此时不包括axis所在的列
    :param dataSet: 待划分的数据集
    :param axis: 划分数据的特征，索引的位置
    :param value: 需要返回的特征的值
    :return: 划分好的数据集
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            '''
           下面两行就是去掉每一行axis的位置，第一行先取axis之前的，第二行再去axis之后的，
           用extends（意思‘+’）连接，因为下一次在决策树划分数据集时已经不需要这个特征了
           '''
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])

            retDataSet.append(reduceFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的特征
    :param dataSet: 待划分数据集
    :return:  划分最好的特征
    '''
    numFeatures = len(dataSet[0])-1          # 特征的个数 eg:dataSet = [[1,1,'yes'],[1,1,'yes'],...]
    baseEntropy = calcShannonEnt(dataSet)

    bestInfGain = 0.0 ; bestFeature = -1

    '''
       大循环是针对特征循环，即索引的数量，对每一维特征进行计算
       小循环是针对每维特征的可能取值计算，即 [0,1]或[0,1,2]
    '''
    for i in range(numFeatures):              # 对不同的特征
        featList = [example[i] for example in dataSet]        # featList = {list}<type 'list'>:[1,1,1,0,0]
        uniqueVals = set(featList)                             # uniqueVals = {set}set([0,1])  set去重，获取所有特征的类别
        newEntropy = 0.0

        '''
        公式：H（Y|X） = E P（X=x）H（Y|x）
        '''
        for value in uniqueVals:                 # 对每个特征的可能取值计算
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfGain):        # 计算最好的信息增益
            bestInfGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''
        多数表决的函数，当递归循环遍历所有的属性时，还无法决策出分类是，采用多数表决的方式决定分类
    :param classList:  待分类的数据集
    :return:  多数表决的结果 sortedClassCount = {'A':2, 'B':1}
    '''
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def creatTree(dataSet, labels):
    '''
    创建树的函数：返回值为创建好的字典树（eg:{'surfing':{0:'no',1:{'flipper':{0:'no',1:'yes'}}}})
    :param dataSet: 数据集
    :param label: 标签列表，算法本身是不需要这个变量的，但是为了对数据给出明确的含义，需要将其作为一个输入的参数提供
    :return: 创建好的树
    '''
    '''
        递归函数的出口
    '''
    classList = [example[-1] for example in dataSet]          # classList中包含了数据集所有的标签
    if classList.count(classList[0]) == len(classList):       # 类别完全相同时，停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:                                  # 遍历完所有的特征时，返回出现次数最多的
        return majorityCnt(classList)

    '''
        递归函数
    '''
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]            # 具体化最优特征

    MyTree = {bestFeatLabel:{}}
    del(labels[bestFeat])            # 删除列表中的之前的最好的标签
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVal = set(featValues)
    for value in uniqueVal:
        '''
        复制了类标签，这是因为在把列表作为参数传递时，参数是按照引用方式传递的，
        为了不改变原始列表的内容，使用新变量
        '''
        subLabels = labels[:]
        MyTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet,bestFeat,value), subLabels)   # 这里的dataSet是经过处理的，不包含之前的特征
    return MyTree


if __name__ == '__main__':
    dataSet,labels = createDataSet()
    ShannonEnt = calcShannonEnt(dataSet)
    print ShannonEnt

    retDataSet = splitDataSet(dataSet,1,1)
    print retDataSet

    bestFeature = chooseBestFeatureToSplit(dataSet)
    print bestFeature

    MyTree = creatTree(dataSet,labels)
    print MyTree






