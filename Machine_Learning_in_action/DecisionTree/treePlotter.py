#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/2/25  10:15
---------------------------
    Author       :  WangKun
    Filename     :  treePlotter.py
    Description  :  可视化决策树
                    1、决策树的主要优点是直观可理解，因此需要通过可视化的方式描述分类的过程和结果
                    2、决策树的绘制过程中，对于坐标的变化，重点关注：
                        plotTree.x0ff = -0.5/plotTree.totalW
                        (plotTree.x0ff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.y0ff)
---------------------------
"""
'''
    在绘制决策树的过程中，有许多的坑，重点关注的点有以下几个：
    1、绘图过程中x，y的坐标的变化：
        1、初始化时取初始位置：plotTree.x0ff = -0.5/plotTree.totalW
           好处是：由于按叶子节点的位置等分切割总长为1的x轴，但是实际上叶子节点的位置（3个叶子节点）是（1/6,1/2,5/6）
           因此我们需要利用（-1/6+1/3）= 1/6，(1/6+1/3) = 1/2, (1/2+1/3) = 5/6,为了在叶子节点的位置上使用
           等差数组表示，因此是初始位置是-1/2*1/3 = 1/6
        2、cntrPt = (plotTree.x0ff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.y0ff)
           plotTree.x0ff是最近绘制的叶子节点的位置，在确定当前节点位置时，只需要确定当前节点有多少个叶子节点，
           根据叶子节点的个数（eg：2），可知当前叶子节点所占的长度numLeafs/plotTree.totalW（eg:2/3），而此时的
           决策节点位于叶子节点的中间，即1/2*numLeafs/plotTree.totalW（eg：1/3）,而由于起始位置并不是0，而是左移了
           半个表格，即1/2/plotTree.totalW（eg：1/6），而还要加上最近绘制的叶子节点的位置plotTree.x0ff
           因此综合起来就是：
                        plotTree.x0ff + [1/2/plotTree.totalW+1/2*numLeafs/plotTree.totalW]
                       =(plotTree.x0ff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.y0ff)

           可以根据如下图思考：
                                    （1/2，1）
                                   /         \
                            （1/6,1/2）     （2/3,1/2）
                                         /          \
                                    （1/2,0）        （5/6,0）

    x轴坐标：（-1/6,0） (0, 0)   （1/3,0）    （2/3,0）     （1，0）
'''

import matplotlib.pyplot as plt


# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细，
# 也可以为：decisionNode = {boxstyle:'sawtooth',fc:'0.8'}
decisionNode = dict(boxstyle='sawtooth',fc='0.8')            # 定义文本框和箭头格式
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeTxt,centerPt,parentPt, nodeType):
    '''
    绘图,绘制带箭头的注解
    :param nodeTxt: 结点描述
    :param centerPt: 结点的中心位置，即箭头的终点
    :param parentPt: 结点的父节点，即箭头的起点
    :param nodeType: 结点类型，即对叶子节点和决策结点分别表示
    :return:
    '''
    createPlot.ax1.annotate(nodeTxt,xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    '''
    绘图的demo
    :return: 绘制好的决策树模型
    '''
    fig = plt.figure(1, facecolor='white')          # 绘制白布背景
    fig.clf()                                        # 清空图的背景
    createPlot.ax1 = plt.subplot(111,frameon = False)    # frameon 表示是否绘制坐标矩形框，True为绘制矩形框
    plotNode('Decision Node', (0.5,0.1), (0.1,0.5), decisionNode)
    plotNode('Leaf Node', (0.8,0.1), (0.3,0.8),leafNode)
    plt.show()

'''
    为了构造注解树，我们需要确定树的层次和树的叶子节点的数量
    这里通过getNumLeafs()获得叶节点的数目,getNumDepth()获得树的深度
'''
def getNumLeafs(MyTree):
    '''
    递归的得到树的叶子的数量
    :param MyTree: 通过决策树生成的树的模型
    :return: 返回叶子的数量
    '''
    numLeafs = 0
    firstStr = MyTree.keys()[0]
    secondDict = MyTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])    # 如果子节点是字典类型，则该节点是决策结点，需要再递归的调用函数
        else:
            numLeafs += 1
    return numLeafs

def getNumDepth(MyTree):
    '''
    获取树的高度
    :param MyTree: 决策树生成的树
    :return:  决策树的深度
    '''
    maxDepth = 0
    firstStr = MyTree.keys()[0]
    secondDict = MyTree[firstStr]
    # 该函数的终值条件是叶子节点，一旦到达叶子节点，则从递归调用的返回，并将计算书深度的变量加1
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            thisDepth = 1 + getNumDepth(secondDict[key])
        else:
            thisDepth = 1
    if thisDepth > maxDepth:
        maxDepth = thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
    '''
    在父节点和子节点之前填充文本信息
    :param cntrPt: 子节点的坐标位置
    :param parentPt: 父节点的坐标位置
    :param txtString: 文本信息
    :return:  父子节点间的文本信息
    '''
    xMid = (parentPt[0] + cntrPt[0])/2
    yMid = (parentPt[1] + cntrPt[1])/2
    createPlot.ax1.text(xMid,yMid,txtString)

'''
    这里是绘制图形最关键的位置，绘制过程的思想是：
    1、绘制自身
    2、如果是叶子节点，则绘制
    3、如果是决策结点，则递归

    绘图的主函数，其主要调用plotTree函数，而plotTree函数又调用getNumLeafs()和getNumDepth()
'''
def createPlot(inTree):
    '''
    绘图的主函数
    :param inTree:  决策树模型
    :return: 绘制的图形
    '''
    fig = plt.figure(1,facecolor='white')
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon = False, **axprops)
    '''
    plotTree.totalW和plotTree.totalD是全局变量，存储的是树的高度和深度，用于plotTree调用的参数
    plot.x0ff和plot.y0ff也是全局变量，这两个变量用于追踪当前的节点位置，以及放置下一个节点的位置
    '''
    plotTree.totalW = float(getNumLeafs(inTree))      # 树的叶子节点的数目
    plotTree.totalD = float(getNumDepth(inTree))      # 树的深度

    '''
    重点理解的地方1：
        初始化时取初始位置：plotTree.x0ff = -0.5/plotTree.totalW
        好处是：
        虽然是按叶子节点的个数等分切割总长为1的x轴，即（1/3,2/3和1），
        但是实际上叶子节点的位置（3个叶子节点）是（1/6,1/2,5/6）
        因此我们需要利用（-1/6+1/3）= 1/6，(1/6+1/3) = 1/2, (1/2+1/3) = 5/6,为了在叶子节点的位置上使用
        等差数组（或者整数倍的1/plotTree.totalW）表示，因此是初始位置是-1/2*1/3 = -1/6
    '''
    plotTree.x0ff = -0.5/plotTree.totalW
    plotTree.y0ff = 1.0

    plotTree(inTree, (0.5,1.0), '')
    plt.show()

def plotTree(MyTree, parentPt, nodeText):
    '''
     绘制决策树模型
    :param MyTree: 决策树模型生成的决策树
    :param parentPt: 父节点
    :param nodeText: 文本内容
    :return:  绘制完成的决策树模型
    '''
    numLeafs = getNumLeafs(MyTree)        # 计算决策树的叶子节点的数量，即宽度
    depth = getNumDepth(MyTree)           # 计算决策树模型的深度，即高度

    firstStr = MyTree.keys()[0]
    '''
    重点理解的位置2：
           cntrPt = (plotTree.x0ff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.y0ff)
    plotTree.x0ff是最近绘制的叶子节点的位置，在确定当前节点位置时，只需要确定当前节点有多少个叶子节点，
    根据叶子节点的个数（eg：3），可知当前叶子节点所占的长度numLeafs/plotTree.totalW（eg:3/3），而此时的
    决策节点位于叶子节点的中间，即1/2*numLeafs/plotTree.totalW（eg：1/2）,而由于起始位置并不是0，而是左移了
    半个表格，即1/2/plotTree.totalW（eg：1/6），而还要加上最近绘制的叶子节点的位置plotTree.x0ff（eg：-1/6）
    因此综合起来就是：
                 plotTree.x0ff + [1/2/plotTree.totalW+1/2*numLeafs/plotTree.totalW]
               =(plotTree.x0ff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.y0ff)

    '''
    cntrPt = (plotTree.x0ff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.y0ff)
    plotMidText(cntrPt, parentPt, nodeText)
    plotNode(firstStr, cntrPt,parentPt, decisionNode)

    secondDict = MyTree[firstStr]
    '''
       按比例减少y的偏移量，eg：树的层次为2，每次减少1/2
    '''
    plotTree.y0ff = plotTree.y0ff - 1.0/plotTree.totalD

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            '''
            如果是叶子节点，则按比例计算x轴的坐标，eg：叶子节点3个，比例为1/3
           '''
            plotTree.x0ff = plotTree.x0ff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.x0ff,plotTree.y0ff),cntrPt,leafNode)
            plotMidText((plotTree.x0ff,plotTree.y0ff),cntrPt,str(key))
    plotTree.y0ff = plotTree.y0ff + 1.0/plotTree.totalD

def retrieveTree():
    '''
    模拟树数据
    :return:模拟树
    '''
    Tree = {'no surfacing':{0:'no', 1:{'flippers':{0:'no', 1:'yes'}}}}
    return Tree

if __name__ == '__main__':
    Tree = retrieveTree()
    # numLeafs = getNumLeafs(Tree)
    # maxDepth = getNumDepth(Tree)
    #
    # print numLeafs
    # print maxDepth
    createPlot(Tree)