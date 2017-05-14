#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/14  11:09
---------------------------
    Author       :  WangKun
    Filename     :  Gui.py
    Description  :  用Python的Tkinter库创建GUI
                    1、真正的用户界面不仅仅是展示一个静态图像，而是用户不需要指令的通过按照自己的方式分析数据，这种能够同时支持
                    数据呈现和用户交互的方式就是构建一个图形用户界面（GUI,Graphical User Interface）。
                    2、步骤：
                        1、利用现有的模块Tkinter来构建GUI
                        2、使用Thinter与绘图库交互
---------------------------
"""
from Tkinter import *
import matplotlib
matplotlib.use('TKAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import *
import cart
import modelTree


def loadDataSet(filename):
    dataMat = []
    with open(filename, 'r')as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            # 高阶内置函数，将每一行映射为浮点数, 对序列中的每一个元素都进行处理，for循环的简化版
            fltLine = map(float, line)
            dataMat.append(fltLine)
    return dataMat

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X * model)


def regTreeEval(model, inDat):
    return float(model)


def treeForeCast(tree,inData,modelEval = regTreeEval):
    if not cart.isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if cart.isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'],inData)
    else:
        if cart.isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)


def createForeCast(tree,testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat


def reDraw(tolS,tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = modelTree.createTree(reDraw.rawDat,ops=(tolS,tolN))
        yHat = createForeCast(myTree,reDraw.testDat,modelTreeEval)
    else:
        myTree = cart.createTree(reDraw.rawDat,ops=(tolS,tolN))
        yHat = createForeCast(myTree,reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:,0],reDraw.rawDat[:,1],s=5)
    reDraw.a.plot(reDraw.testDat,yHat, linewidth=2.0)
    reDraw.canvas.show()

def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print 'enter Inter for tolN'
        tolNentry.delete(0,END)
        tolNentry.insert(0,'10')
    try:
        tolS = int(tolSentry.get())
    except:
        tolS = 1.0
        print 'enter Inter for tolS'
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN,tolS


def drawNewTree():
    tolN,tolS = getInputs()
    reDraw(tolS,tolN)

root = Tk()

reDraw.f = Figure(figsize=(5,4),dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f,master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0,columnspan =3)

Label(root,text='tolN').grid(row=1,column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1,column=1)
tolNentry.insert(0,'10')

Label(root,text='tolS').grid(row=2,column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2,column=1)
tolSentry.insert(0,'1.0')

Button(root,text='Redraw',command=drawNewTree).grid(row=2,column=2,rowspan=3)

chkBtnVar = IntVar()
chkBtn = Checkbutton(root,text='Model Tree',variable=chkBtnVar)
chkBtn.grid(row=3,column=0,columnspan=2)

reDraw.rawDat = mat(loadDataSet('sine.txt'))
reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0,10)

root.mainloop()

'''
GUI分析：
    1、tolS：容许的误差下降值，tolN：切分的最小样本数（二分后两侧的样本数量）
    2、回归树：
        1、tolS越大时（tolN不变），则容许的误差下降值越大，即只会出现总方差改变较大的分割，如果总方差改变较小，则不会
    分割，也即分割的效果越不好。eg：在chooseBestSplit()函数中，if （S - bestS） < 10 和 if （S - bestS） < 1，当
    （S - bestS） = 8时，按照第一种是不会分割的，而按照第二种则会分割。也即原本可以分割的现在不满足条件了不能分割，也
    即第一种时要求的误差下降必须要很大才会分割，故分割的也越粗糙，分割效果也越不好
        2、tolN越大时（tolS不变），则切分后的两侧样本数量越多，分割之后的数据集较大时也不再分割，即不再进行进一步二分，
    则切分效果越不好。
    3、模型树：
        1、tolS越大时（tolN不变），同上，tolS越大二分效果越不好
        2、tolN越大时（tolS不变），同上，tolN越大二分效果越不好
'''

