#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/21  20:00
---------------------------
    Author       :  WangKun
    Filename     :  pic_dimen_redu.py
    Description  :  SVD矩阵分解用于图像压缩
---------------------------
"""
from numpy import *


def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print 1,                    # 在输出print后加, 表示不换行
            else:
                print 0,
        print ''


def imgCompress(numSv=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)

    myMat = mat(myl)
    # print myMat
    print '**********original matrix*************'
    printMat(myMat, thresh)
    U, sigma, V = linalg.svd(myMat)
    SigRecon = mat(zeros((numSv, numSv)))

    for k in range(numSv):
        SigRecon[k, k] = sigma[k]
    reconMat = U[:,:numSv] * SigRecon * V[:numSv,:]                     # 恢复矩阵
    print '*******reconstructed matrix using %d singular values***************' % numSv
    printMat(reconMat, thresh)

if __name__ == '__main__':
    imgCompress(2)