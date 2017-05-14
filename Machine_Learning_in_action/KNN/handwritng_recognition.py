#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/2/21  10:30
---------------------------
    Author       :  WangKun
    Filename     :  KNN-handwriting_recogniton.py
    Description  :  k-nearest neighbor用于手写识别系统
                    1、32 * 32维的图像，可以将其二进制图像转换成为1 * 1024维的向量，这里再利用KNN算法，处理数字图像信息
                    2、步骤：
                       1、将图形转换为向量
                       2、测试，这里涉及到从文件夹中读取所有的文件，需要导入os模块的listdir
---------------------------
"""

from numpy import *
from os import listdir
import KNN


def img2vector(filename):       # 将图像转化为向量
    returnVector = zeros((1,1024))
    with open(filename,'r') as f:
        for i in range(32):
            lineStr = f.readline()
            for j in range(32):
                returnVector[0,32*i+j] = int(lineStr[j])
    return returnVector


def handwritingClassTest():
    hwLabel = []
    trainingFileList = listdir('trainingDigits')      # 获取目录内容 ，type（list）
    m = len(trainingFileList)

    trainingMat = zeros((m,1024))
    for i in range(m):
        filenameStr = trainingFileList[i]
        filename = filenameStr.split('.')[0]
        classNum = int(filename.split('_')[0])
        hwLabel.append(classNum)                       # 从文件名中解析分类数字
        trainingMat[i,:] = img2vector('trainingDigits/%s' % filenameStr)

    testFileList = listdir('testDigits')
    m_test =len(testFileList)
    error_count = 0.0
    for j in range(m_test):
        test_filenameStr = testFileList[j]
        test_filename = test_filenameStr.split('.')[0]
        test_ClassNum = int(test_filename.split('_')[0])            # 通过文件名获取实际的数字编号
        classfierResult = KNN.classify(img2vector('testDigits/%s'% test_filenameStr),trainingMat,hwLabel,4) # 通过KNN算法得到的编号

        print 'the classfier came back with: %d, the realnum came back with %d' %(classfierResult, test_ClassNum)
        if classfierResult != test_ClassNum:
            error_count += 1
    print 'the total error num: %d' % error_count
    print 'the total error rate is :%f' % (error_count/m_test)


if __name__ ==  '__main__':
    # returnVector = img2vector('0_5.txt')
    # print returnVector[0, 0:31]
    handwritingClassTest()











