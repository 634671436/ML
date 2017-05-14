#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/13  10:23
---------------------------
    Author       :  WangKun
    Filename     :  cart.py
    Description  :  树回归
                    1、树回归：
                        将数据集切分为很多份易建模的数据，然后利用线性回归进行建模和拟合
                        与线性回归的不同点：线性回归需要拟合所有的样本点，当数据众多且特征之间关系十分复杂时，这种想法就很难
                    拟合好数据，而且在实际问题中，大多数可能都是非线性的，eg：分段函数。不可能使用全局模型去拟合任何数据。
                    2、回归树与模型树：
                        chooseBestSplit()函数，如果构建的是回归树，则返回的模型为常数；如果构建的是模型树，返回的则是线性方
                    程。
                    3、CART用于回归：
                        回归树假设叶子节点的值为常数。
                        如何度量数据一致性（也就是如何切分数据）：计算数据的均值，然后计算每条数据到均值的差值，这里用总方差
                    度量。
                    4、树剪枝：
                        树剪枝是为了避免过拟合的出现，但是为了判断是否出现过拟合，需要在测试集上使用交叉验证。
                        预剪枝：在chooseBestSplit()函数中其实进行了一部分的预剪枝工作，也就是tolS和tolN的人为设定，但是这种
                    需要人工设定的初始值往往随着数量级的增加而花费大量的时间去尝试，这种效果并不好。因此用到后剪枝。
                        后剪枝：运用交叉验证集，将数据集分为测试集和验证集，首先通过指定tolS和tolN，构建出足够大的树，使其足够
                    复杂，便于剪枝。接下来从上而下找到结点，用测试集来判断将这些叶节点合并是否能降低测试误差。
---------------------------
"""
'''
个人理解：CART与ID3的区别：
        1、ID3的决策树在实行的过程中，如果一旦按照某种特征进行切分之后，该特征在之后的算法过程中将不会再起作用，
    也即人们认为的该算法切分过于迅速，而CART的每一个结点的选择和分类都基于所有的特征进行选择，符合条件的进入左子
    树（右子树），不符合条件的进入右子树（左子树）。
'''

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(filename):
    dataMat = []
    with open(filename, 'r')as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            # 高阶内置函数，将每一行映射为浮点数, 对序列中的每一个元素都进行处理，for循环的简化版
            fltLine = map(float, line)
            dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    '''
    二元切分数据集
    :param dataSet: 待切分数据集
    :param feature: 待解分的特征
    :param value: 该特征的某个值
    :return: 切分好的数据集
    '''
    '''
    nonzero()函数以元组方式返回满足条件的坐标目录，所以这里的nonzero()[0]返回的是所有满足条件的行的坐标
    '''
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    '''
        创建叶节点的函数,当chooseBestSplit()函数确定不再对数据进行切分时，
    将调用该函数来得到叶子节点的模型。在回归树时就是均值。
    :param dataSet:
    :return:
    '''
    return mean(dataSet[:, -1])


def regErr(dataSet):
    '''
    按数据集计算总方差
    :param dataSet:
    :return:
    '''
    return var(dataSet[:, -1]) * shape(dataSet)[0]
'''
createTree()伪代码：     ---迭代函数
找到最佳的待切分特征：
    如果该结点不能再分，将该节点存为叶子节点
    执行二次切分
    在右子树调用createTree()
    在左子树调用creatrTree()
'''
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    树构造函数
    :param dataSet: 数据集
    :param leafType: regLeaf建立叶节点函数
    :param errType: regErr 代表误差计算函数
    :param ops:
    :return:
    '''
    '''
    chooseBestSplit()函数，如果构建的是回归树，则返回的模型为常数；如果构建的时模型树，返回的则是线性方程
    '''
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:  # 满足停止条件时，返回叶子节点值，这里也只递归的出口
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


'''
算法的精华部分：
    对每个特征：
        对每个特征值（可能取值）：
            将数据集切分为两部分
            计算切分误差（这里用到的是总方差）
            如果当前误差小于当前最小误差，那么将当前误差设定为最佳切分并更新最小误差
    返回最佳切分的特征和阈值
'''
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 用户设定的参数，用于控制函数的停止时机，具体的tolS是容许的误差下降值；tolN是切分的最小样本树
    tolS = ops[0]
    tolN = ops[1]

    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 如果所有的值都相等，则退出
        return None, leafType(dataSet)

    m, n = shape(dataSet)
    S = errType(dataSet)  # 当前数据集的总方差
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):  # 意即：选择原来样本中的每一维特征的每一种可能取值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 如果切分出来的数据集很小，就不应该进行切分
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:  # 如果切分数据集之后的方差改变不大，则直接生成叶子，以均值作为叶子节点的输出
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def isTree(obj):
    return type(obj).__name__=='dict'


def getMean(tree):
    '''
    从上往下遍历，直到叶子节点为止，如果是叶子节点则返回均值。递归的求整棵树的均值
    :param tree:生成的树
    :return:
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0


def prune(tree, testData):
    '''

    :param tree: 训练树
    :param testData: 测试数据集
    :return: 返回剪枝后的数据集
    '''
    if shape(testData)[0] == 0:  # 如果没有测试数据，则返回树的均值，也叫作塌陷处理
        return getMean(tree)

    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    # 如果两个分支不再是子树，就合并他们
    # 具体的做法如下：
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))

        # 对比合并前后的误差，如果合并后的误差比不合并的小就进行合并操作，反之则不合并直接返回
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))

        if errorMerge < errorNoMerge:
            print 'merging'
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__ == '__main__':
    '''
    TEST
    '''
    # dataMat = loadDataSet('ex0.txt')
    # print dataMat

    # testMat = mat(eye(4))
    # print testMat

    # 按元祖返回满足条件的元素的坐标目录
    # t1 = nonzero(testMat[:,1] <= 0.5)
    # t2 = nonzero(testMat[:, 1] <= 0.5)[0]
    # print t1                # eg：(array([0, 2, 3], dtype=int64), array([0, 0, 0], dtype=int64))，前行后列
    # print t2                # eg:  [0 2 3]
    # print testMat[[0,2,3],:]   # 返回第0行、第2行、第3行的数据

    # mat0,mat1 = binSplitDataSet(testMat,1,0.5)
    # print mat0,mat1

    '''
    SPLIT1
    '''
    # xList = []
    # yList = []
    # with open('ex00.txt','r')as f:
    #     for line in f.readlines():
    #         line = line.strip().split('\t')
    #         xList.append(line[0])
    #         yList.append(line[1])
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.scatter(xList, yList)
    #     ax.scatter(0.48813,1.0180967672413792,color='red')     #模拟切分点处左子树的点，即左子树的输入（均值）
    #     ax.scatter(0.48813, -0.044650285714285719, color='blue')  #模拟切分点处右子树的点，即右子树的输入（均值）
    #     plt.show()

    # myDat = loadDataSet('ex00.txt')
    # myMat = mat(myDat)
    # retTree = createTree(myMat)
    # print retTree

    '''
    SPLIT3
    '''
    # xList = []
    # yList = []
    # with open('ex0.txt','r')as f:
    #     for line in f.readlines():
    #         line = line.strip().split('\t')
    #         xList.append(line[1])
    #         yList.append(line[2])
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.scatter(xList, yList)
    #     plt.show()
    #
    # myDat = loadDataSet('ex0.txt')
    # myMat = mat(myDat)
    # retTree = createTree(myMat)
    # print retTree

    '''
    SPLIT4
    '''
    myDat = loadDataSet('ex2.txt')
    myMat = mat(myDat)
    myTree = createTree(myMat, ops=(0, 1))
    # print myTree

    myDatTest = loadDataSet('ex2test.txt')
    myDatTest = mat(myDatTest)
    # print myDatTest

    NewTree = prune(myTree, myDatTest)
    print NewTree






