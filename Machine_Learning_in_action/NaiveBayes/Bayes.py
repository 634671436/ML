#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/2/27  15:36
---------------------------
    Author       :  WangKun
    Filename     :  Bayes.py
    Description  :  NaiveBayes.py
                    朴素贝叶斯：基于贝叶斯定理和特征条件独立假设的分类方法。
                        1、贝叶斯定理：
                                P(Y|X) = P(X,Y)/P(X)
                                       = P(X|Y)P(Y)/P(X)
                                P(X|Y)，P(Y)可以通过试验样本计算，其中P(Y)为先验概率
                            注意：利用样本计算先验概率是一种极大似然估计或者贝叶斯估计，先验概率是一个独立于实验样本存在的确定
                            的值，实验中由于无法确知该值，因此利用实验中的概率值去拟合估计该值
                        2、‘朴素’
                               朴素贝叶斯对条件概率分布做了条件独立的假设
                               P(X|Y) = P(X1 ,X2, X3, X4....|Y)
                                 本来：= P(X1|Y)P(X2|X1，Y)P(X3|X2，X1，Y)...
                                 实际：= P（X1|Y）*P(X2|Y)*P(X2|Y)...
                        3、由于对于一个样本而言，P(X)是相同的，因此在比较归属样本时，只用比较分子即可
                        4、典型应用：
                            1、文档分类，整个文档是一个实例，其中电邮的某些元素构成了特征，常用的特征是文档中出现的词
                                首先是拆分文本，特征是来自文本的词条（token），可以理解为单词，或者是非单词词条，如URL、IP地址
                            或者其他任意的字符，那么每一个文本片段都可以表示为一个向量，其中值1表示出现过，0表示没出现
---------------------------
"""
from numpy import  *


'''
   朴素贝叶斯算法的python实现过程：
   1、首先获取文本数据dataSet
   2、文本数据通过createVocabList(dataSet)函数创建词典列表
   3、文本数据根据词典列表创建成向量，setOfWord2Vector(vocabList,dataSet),其中向量长度为词典列表长度
   4、计算P(x1|y=1)、P(x1|y=0)...，通过trainNBO(trainMatrix,trainCategory)，trainMatrix就是第三步中所有的向量append
   5、将P(Y=1)P(X1=1|Y=1)P(X2=1|Y=1)转化为
            log[P(Y=1)P(X1=1|Y=1)P(X2=1|Y=1)]
           =logP(Y=1)+log(P(X1=1|Y=1))+logP(X2=1|Y=1),这样第四步中的就能直接相加
'''


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'please', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'is', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]  # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec


def createVocabList(dataSet):
    '''
    创建一个包含所有文档中不重复出现的词的列表
    :param dataSet:
    :return:
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)         # 两个集合的并集
    return list(vocabSet)

'''
    1、文档存在两种形式的表现，第一是词集（set-of-words model）模型，一种是词袋模型(bag-of-words model)
        词集（set-of-words model）模型：
            每个单词只能出现一次
        词袋模型(bag-of-words model)模型：
            每个单词根据出现的次数叠加，可以出现多次
'''
def setOfWord2Vector(vocabList,dataSet):
    '''
    将输入的字词列表转化成向量,向量的长度与字典列表的顺序一样
    :param vocabList:  字典列表
    :param inputSet:   输入的字词
    :return:
    '''
    returnVec = [0]*len(vocabList)

    for word in dataSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1   # vocabuList.index(word)返回的是vocabuList中word对应的索引
    return returnVec

'''
  词袋模型(bag-of-words model)模型：
'''
def bag_of_words(vocabList,inputSet):
    returnVec = [0]*len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

'''
    未优化贝叶斯分类器训练函数
'''
def trainNBO(trainMatrix,trainCategory):
    '''
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 训练矩阵
    :param trainCategory: 训练文档类别的向量
    :return:
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory)/float(numTrainDocs)               # 计算先验概率P(Y)

    p0Num = zeros(numWords);p1Num = zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]                # 矩阵相加
            # p1Denom += sum(trainMatrix[i])
            p1Denom += 1
        else:
            p0Num += trainMatrix[i]
            # p0Denom += sum(trainMatrix[i])
            p0Denom += 1
    p1Vect = p1Num/p1Denom
    # p1Vect = p1Num/sum(trainCategory)
    p0Vect = p0Num/p0Denom
    # p0Vect = p0Num/float(numTrainDocs-sum(trainCategory))
    return pAbusive,p1Vect,p0Vect

'''
    对以上函数的优化，3处改变
    1、为避免存在P(x|y) = 0的情况，则将所有单词出现的起始值设为1
    2、同时在计算p0Num、p1Num时，为避免存在概率大于1的情况，将p0Denom、p1Denom设置为样本总数
    3、在计算中，由于单词库较大，p(x1|1)p(x2|1)...趋向于0，因此取对数，也为了之后计算方便
'''
def TrainingNB1(trainMatrix,trainCategory):
    '''
    修改后的trainNBO(trainMatrix,trainCategory)函数
    :param trainMatrix: 训练矩阵
    :param trainCategory: 训练文档类别的向量
    :return:
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算先验概率

    p0Num = ones(numWords)    # 为了避免存在p(x1|1)p(x2|1)...中等于0的情况，这里改为ones（numWord）,以下同理
    p1Num = ones(numWords)

    p0Denom = numTrainDocs     # 改变2
    p1Denom = numTrainDocs

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            # p1Denom += sum(trainMatrix[i])
            p1Denom += 1
        else:
            p0Num += trainMatrix[i]
            # p0Denom += sum(trainMatrix[i])
            p0Denom += 1
    p1Vect = log(p1Num / p1Denom)           # 改为log
    # p1Vect = p1Num/sum(trainCategory)
    p0Vect = log(p0Num / p0Denom)           # 改为log
    # p0Vect = p0Num/float(numTrainDocs-sum(trainCategory))
    return pAbusive, p1Vect, p0Vect

'''
    这里可以看到，我们在用实例进行测试的时候，只需要将文字转化为向量，就能直接计算
'''
def testingNB():
    '''
    :return: 直接返回分类的结果
    '''
    postingLists, classVec = loadDataSet()
    MyVocabulist = createVocabList(postingLists)

    trainMarix = []
    for postDoc in postingLists:
        trainMarix.append(setOfWord2Vector(MyVocabulist, postDoc))

    pAb,p1v,p0v = TrainingNB1(array(trainMarix),array(classVec))

    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWord2Vector(MyVocabulist,testEntry))

    '''
       通过ln（a*b） = lna + lnb将乘法改为相加的运算
       log(P(Y=1)P(X1=1|Y=1)P(X2=1|Y=1)) = log(P(Y=1))+log(P(X1=1|Y=1))+logP(X2=1|Y=1))
       而log(P(X1=1|Y=1))和logP(X2=1|Y=1))在TrainingNB1()函数中就已经计算了
    '''
    p1 = sum(p1v * thisDoc) + log(pAb)
    p0 = sum(p0v * thisDoc) + log(1-pAb)

    if p1 > p0:
        print testEntry,'classified as: 1'
    else:
        print testEntry,'classified as: 0'


if __name__ == '__main__':
    postingLists, classVec = loadDataSet()
    MyVocabulist = createVocabList(postingLists)
    print MyVocabulist
    # print setOfWord2Vector(MyVocabulist,postingList[0])

    # trainMarix = []
    # for postDoc in postingLists:
    #     trainMarix.append(setOfWord2Vector(MyVocabulist,postDoc))
    #
    # pAbusive, p1Vect,p0Vect = trainNBO(trainMarix,classVec)
    # print pAbusive
    # print p1Vect
    # print p0Vect
    testingNB()
















