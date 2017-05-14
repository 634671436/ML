#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/2/28  10:04
---------------------------
    Author       :  WangKun
    Filename     :  spamEmail.py
    Description  :  NaiveBayes.py
                    1、过滤垃圾邮件
---------------------------
"""
import random
import re
from numpy import *
import Bayes

def textParse(bigString):
    '''
    文件解析
    :param bigString: 待分割的字符串
    :return:
    '''
    pattern = re.compile(r'\W')
    listOfTokens = re.split(pattern,bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamDict():

    docList = []        # 切分好的词组成的列表[['his','xx','xx'],['xx','xx','xx'],['xx','xx','xx']]
    classList = []      # 邮件的类别，垃圾邮件和正常邮件，垃圾邮件为1，正常邮件为0
    for i in range(1, 26):
        with open('ham/%d.txt' % i) as f:
            wordList = textParse(f.read())
            docList.append(wordList)
            classList.append(1)
        with open('spam/%d.txt'% i) as f:
            wordList = textParse(f.read())
            docList.append(wordList)
            classList.append(0)
    vocabList =Bayes.createVocabList(docList)      # 将docList组成词典

    '''
      从50封电子邮件中随机选出10封作为测试集，剩下的作为训练集
    '''
    trainingSet = range(50)
    testSet = []

    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]

    trainMarix = []             # 训练集数据组成的训练矩阵
    trainingClass = []          # 训练数据集中的类别
    for docIndex in trainingSet:
        trainMarix.append(Bayes.setOfWord2Vector(vocabList,docList[docIndex]))
        trainingClass.append(classList[docIndex])
    pAb, p1v, p0v = Bayes.TrainingNB1(array(trainMarix), array(trainingClass))

    errorCount = 0.0
    for docIndex in testSet:
        thisDoc = array(Bayes.setOfWord2Vector(vocabList, docList[docIndex]))

        if classifyNB(array(thisDoc),p0v,p1v,pAb) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is :', float(errorCount)/len(testSet)


def classifyNB(vec2Classify,p0Vec,p1Vec,pclass):
    p1 = sum(vec2Classify * p1Vec) + log(pclass)
    p0 = sum(vec2Classify * p0Vec) + log(1-pclass)
    if p1 > p0:
        return 1
    if p1 < p0:
        return 0


if __name__ == '__main__':
    # str = 'This book is the best book On python or M.L.I have ever laid eyes upon.'
    # print textParse(str)

    # docList,classList = spamDict()
    # print docList
    # print classList

    vocabList = spamDict()
    print vocabList