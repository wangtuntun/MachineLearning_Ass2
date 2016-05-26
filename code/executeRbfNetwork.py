#coding=utf-8
from rbfNetwork import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import *
import sys
'''
Created on 2016��5��17��

@author: pangb
'''
#运行交叉校验求的较优的k值
def crossKNormal():
    xArr, yArr= RBFNN.loadData('data.txt')
    xArr = regularize(mat(xArr)).tolist()
    errlist = []
    for i in range(10):
        n=RBFNN(2, i+3, 1)
        err = n.rbfCross(xArr[0:2500][:], yArr[0:2500])
        print err
        errlist.append(err)
    
    x1 = [3,4,5,6,7,8,9,10,11,12]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x1, errlist)
    plt.show()

def crossKNoNormal():
    xArr, yArr= RBFNN.loadData('data.txt')
    #xArr = regularize(mat(xArr)).tolist()
    errlist = []
    for i in range(10):
        n=RBFNN(2, i+3, 1)
        err = n.rbfCross(xArr[0:2500][:], yArr[0:2500])
        print err
        errlist.append(err)
    
    x1 = [3,4,5,6,7,8,9,10,11,12]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x1, errlist)
    plt.show()
    
def drawKmeans():
    xArr, yArr= RBFNN.loadData('data.txt')
    clf = KMeans(n_clusters=12)
    xArr1 = xArr[0:2500][:]
    xArr1 = mat(xArr1)
    s = clf.fit(xArr1)
    cents = clf.cluster_centers_
    labels = clf.labels_
    colors = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<y', 'pr', '+b','oy']
    for i in range(12):
        index = nonzero(labels==i)[0]
        x0 = xArr1[index,0]
        x1 = xArr1[index,1]
        for j in range(len(x0)):
            plt.plot(x0[j],x1[j],colors[i],markersize = 12)
        plt.plot(cents[i, 0], cents[i, 1], colors[i],markersize = 12)
    plt.show()

def testPredict():
    xArr, yArr= RBFNN.loadData('data.txt')
    xArrTrain = xArr[0:2500][:]
    yArrTrain = yArr[0:2500]
    xArrTest = xArr[2500:3000][:]
    yArrTest = yArr[2500:3000]
    rbf = RBFNN(2,12,1)
    yPredict = rbf.rbfPredict(xArrTrain, yArrTrain, xArrTest)
    print rbf.calculateDiff(yPredict.tolist(), yArrTest)
    yPredictArr = yPredict.tolist()
    x1 = range(500)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, yPredictArr[0])
    plt.show()
if __name__ == '__main__':
    for arg in sys.argv:
        if arg == '-crossNormal':
            crossKNormal()
        if arg == '-crossNoNormal':
            crossKNoNormal()
        if arg == '-drawKmeans':
            drawKmeans()
        if arg == '-predict':
            testPredict()

    #crossKNoNormal()
    #drawKmeans()
    #crossKNormal()
    #testPredict()
    