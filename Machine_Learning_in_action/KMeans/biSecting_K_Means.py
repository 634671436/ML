#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
    17/3/15  14:14
---------------------------
    Author       :  WangKun
    Filename     :  biSecting_K_Means.py
    Description  :  二分K-均值算法
                    算法思想：首先将所有的点看做是同一个簇，然后将簇一分为二。之后选择其中一个簇继续划分，选择哪一个簇继续划分
                取决于能否最大程度的降低SSE值。上述的过程不断的重复，直到满足用户指定的簇。
---------------------------
"""
from numpy import *
import KMeans


def biKmeans(dataSet, k, distMeans = KMeans.disEclud):
    m = shape(dataSet)[0]
    n = shape(dataSet)[1]
    clusterAssment = mat(zeros((m,n)))
    centroid0 = mean(dataSet,axis=0).tolist()[0]
    centList = [centroid0]    # <type:  'list'>

    for j in range(m):
        clusterAssment[j,1] = distMeans(mat(centroid0),dataSet[j,:]) ** 2

    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            # 划分之后的误差
            pstInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]
            centriodMat, splitClusAss = KMeans.Kmeans(pstInCurrCluster,2,distMeans)
            sseSplit = sum(splitClusAss[:,1])

            sseNoSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])
            print 'sseSplit, and not Split:',sseSplit,sseNoSplit

            if sseSplit + sseNoSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centriodMat
                bestClusAss = splitClusAss.copy()
                lowestSSE = sseSplit + sseNoSplit
        bestClusAss[nonzero(bestClusAss[:,0].A ==1)[0],0] = len(centList)
        bestClusAss[nonzero(bestClusAss[:,0].A ==0)[0],0] = bestCentToSplit

        print 'the bestCentToSplit is:',bestCentToSplit
        print 'the len of bestClusAss is:', len(bestClusAss)

        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClusAss

    return mat(centList), clusterAssment

if __name__ == '__main__':
    data = mat(KMeans.loadDataSet('testSet2.txt'))
    centList, myNewAssment = biKmeans(data,3)
    print centList
    print myNewAssment
















