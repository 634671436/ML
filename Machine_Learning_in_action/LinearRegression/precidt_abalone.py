#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/8  19:38
---------------------------
    Author       :  WangKun
    Filename     :  pedict_abalone.py
    Description  :  线性回归
                    1、线性回归的损失函数是平方损失函数，L(Y,f(X)) = (Y-f(X))^2

                    局部加权线性回归(LWLR)：
                    1、在线性回归发生欠拟合的时候，在估计中引入一些偏差，降低预测的均方误差。
                    2、我们给待预测的点附近的点一定的权重，而使得远离它的点权重较低
                    3、非参数学习方法：
                        1、有参数学习方法是啥？eg：LR。在训练完所有数据之后得到一系列训练参数，然后根据训练参数来预测样本的值，
                    这时不再依赖之前的训练数据，参数是确定的
                        2、非参数学习方法：eg：LWLR。在预测新样本值时每次都会重新训练新的参数，也就是每次预测新的样本值都会依
                    赖训练数据集合，所以每次的参数是不确定的。
                    4、局部加权线性回归的缺点：
                        对每个点做预测时都必须使用整个数据集，因此当训练容量过大时，非参数学习算法需要占用更多的存储容量，计算
                    速度较慢
---------------------------
"""
import regression


def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

if __name__ == '__main__':
    abX, abY = regression.loadDataSet('abalone.txt')
    yHat01 = regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
    yHat1 = regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
    yHat10 = regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)

    regError01 = rssError(abY[100:199],yHat01)
    regError1 = rssError(abY[100:199], yHat1)
    regError10 = rssError(abY[100:199], yHat10)

    print regError01
    print regError1
    print regError10












