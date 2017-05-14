#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/2/27 11:10
---------------------------
    Author       :  WangKun
    Filename     :  lenses.py
    Description  :  隐形眼镜数据集
                    1、通过决策树模型，在包含患者眼部状况的观察条件及医生推荐的隐形眼镜数据集中，
                     帮助患者选择包括硬材质、软材质和不适合佩戴眼镜的结果
---------------------------
"""
import decisiontree
import treePlotter

if __name__ == '__main__':
    with open('lenses.txt')as f:
        lenses = []
        for inst in f.readlines():
            lense = inst.strip().split('\t')
            lenses.append(lense)
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    lenseTree = decisiontree.creatTree(lenses,lensesLabels)
    print lenseTree

    treePlotter.createPlot(lenseTree)
