#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/9  14:58
---------------------------
    Author       :  WangKun
    Filename     :  ridge_regression.py
    Description  :  线性回归
                    1、线性回归的损失函数是平方损失函数，L(Y,f(X)) = (Y-f(X))^2
---------------------------
"""
'''
    线性回归的个人理解：
    1、岭回归（在回归分析中也被称作为'正则化'，L2正则化）:
            从《机器学习实战》中可以看到，如果特征比样本点还多（n > m）时，也就是说输入数据的矩阵X不是满秩矩阵（满秩矩
        阵是求逆矩阵的充分必要条件，一般的回归问题中X通常是列满秩，而当n > m时，不可能是列满秩。当不能满足列满秩时，也
        就不能对（X^T * X）^ I 求逆。）
            通俗的理解上来说，当n > m时，没有过多的数据（训练集）去约束变量，因此容易出现过拟合的现象。从这个意义上讲
        岭回归的出现是为了解决过拟合的现象。
            那么在线性回归的拟合过程中，局部加权线性回归是为了解决欠拟合，而岭回归则是为了解决过拟合。
    2、正则化的分类
        正则化分为L0正则、L1正则（eg：lasso）、L2正则（eg：岭回归）
        1、L0正则：
            L0正则化对应于0向量范数，向量的0范数是指向量中非零元素的个数，L0正则化的值是模型中非零参数的个数，L0正则化
        可以实现模型参数的的稀疏化。模型参数稀疏化使得模型能自动的选择比较重要的特征属性进行y(i)的预测，去掉没用的信息
        项。但是由于LO正则化是个NP问题，很难求解，这才引出L1正则化和L2正则化
        2、L1正则：L1范数是指向量各元素绝对值之和，美称为'稀疏规则算子'。
            L1正则化中通过稀疏模型参数来降低模型复杂度，可以从两个方面做解读：
            1、L1正则是L0正则'最优凸近似'，而L0正则化是选择比较重要的特征，而丢掉没有信息的项
            2、参考文末的L1和L2的区别
        3、L2正则（岭回归）：
            1、《机器学习实战》一书中用缩减系数来描述L2正则，L2正则通过减少模型参数（w）来控制过拟合的效果，在损失函数
        中加入lambda * （系数平方和）。
            NG的课程中提到，在正则化里我们要做的事情，就是减小我们的代价函数所有的参数值（w）以达到能够减小不重要的参
        数，并且让代价函数最优化的函数来选择这些惩罚的程度，因为我们并不知道是哪一个或哪几个要去缩小。因此，我们需要修
        改写代价函数，当我们添加一个额外的正则化项的时候，我们收缩了每个参数。

            那么为什么减小了参数就能避免过拟合呢？实际上，这些参数的值越小，通常对应于越光滑的函数，也就是更加简单的函
        数。因此就不易发生过拟合的问题,从而提高模型的返回能力。

            2、现在讨论lambda取值的作用：要做的就是控制在两个不同的目标中的平衡关系。第一个目标就是我们想要训练，使假
        设更好地拟合训练数据。我们希望假设能够很好的适应训练集。而第二个目标是我们想要保持参数值较小。而lambda这个正则
        化参数需要控制的是这两者之间的平衡，即平衡拟合训练的目标和保持参数值较小的目标。从而来保持假设的形式相对简单，
        来避免过度的拟合。

            3、接着讨论L2正则化中lambda的取值的影响，首先明确方差指的是模型之间的差异（也即用不同的样本集针对同一个
        模型训练），而偏差指的是模型预测值与数据之间的差距。
            根据岭迹图的观察，可以发现当lambda在很小的时候，也即正则化项不起较大作用的时候，参数和线性回归相同。而当
        lambda非常大的时候，为了最小化代价函数，则参数值都趋向0。这提示我们需要在这个变化过程中某处找到使得预测的结果
        最好的lambda的值。
            对应着看，当lambda很小的时候，模型的复杂度还是较高的，这个时候是低偏差高方差。而当lambda很大的时候，模型
        的复杂度降低，过拟合的现象缓解，此时是高方差低偏差。

        4、L1正则化和L2正则化的差别和联系：
            1、直观上的解释。想象y1 = a|x| 和 y2 = ax^2，对这两个函数求导的时候，在相同的a的情况下，y1会比y2在趋向
        于0的时候下降的更快，因此会更快的下降到0。
            2、通俗的解释是：L1会趋向于产生少量的特征，而其他的特征都是0；而L2会选择更多的特征，这些特征都会接近于0。
        Lasso在特征选择时候非常有用，而Ridge就只是一种规则化而已。
'''



from numpy import *
import regression
import matplotlib.pyplot as plt


def ridgeRegress(xMat,yMat,lam=0.2):
    '''
    用于计算回归系数
    :param xMat:输入矩阵
    :param yMat: 标签矩阵
    :param lam: 参数
    :return:
    '''
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print 'This matrix is singular,connot do inverse'
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T

    # 做数据标准化操作，具体的操作为：所有的特征都减去各自的均值并除以方差
    # numpy.mean(, axis = 0 / 1)   0是按列求均值，1是按行求均值
    # numpy.var( axis = 0)    按列求方差
    yMean = mean(yMat,0)  # yMean = [[9.93368446]]
    yMat = yMat - yMean

    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat - xMeans)/xVar

    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegress(xMat, yMat, exp(i-10))
        wMat[i,:] = ws.T
    return wMat

if __name__ == '__main__':

    abX, abY = regression.loadDataSet('abalone.txt')
    ridgeWights = ridgeTest(abX, abY)

    print ridgeWights

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWights)
    plt.show()







