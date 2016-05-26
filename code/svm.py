#coding=utf-8
'''
Created on 2016��5��17��

@author: pangb
'''
from sklearn import svm
import numpy
def regularize(xMat):
    #inMat = xMat.copy()
    inMeans = numpy.mean(xMat, 0)
    inVar = numpy.var(xMat, 0)
    xMat = (xMat - inMeans) / inVar

    return xMat
def loadData(fileName):
    numFeat = len(open(fileName).readline().strip().split(' ')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split(' ')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat
def diff(y1, y2):
    count = 0
    for i in range(len(y1)):
        if abs(y1[i] - y2[i]) < 1:
            count += 1
    return count
            
def linearSVM():
    xArr, yArr = loadData('data.txt')
    xArr = regularize(xArr).tolist()
    xArrTrain = xArr[0:2500][:]
    yArrTrain = yArr[0:2500]
    xArrTest = xArr[2500:3000]
    yArrTest = yArr[2500:3000]
    linear_svc = svm.SVC(kernel='linear')
    linear_svc.fit(xArrTrain, yArrTrain)
    yPredict = linear_svc.predict(xArrTest)
    #print yPredict
    print '正确的个数:'
    num = diff(yPredict, yArrTest)
    print num
    print '正确率'
    print float(num) / len(yArrTest)

def rbfSVM():
    xArr, yArr = loadData('data.txt')
    #xArr = regularize(xArr).tolist()
    xArrTrain = xArr[0:2500][:]
    yArrTrain = yArr[0:2500]
    xArrTest = xArr[2500:3000]
    yArrTest = yArr[2500:3000]
    linear_svc = svm.SVC(kernel='rbf')
    linear_svc.fit(xArrTrain, yArrTrain)
    yPredict = linear_svc.predict(xArrTest)
    #print yPredict
    print '正确的个数:'
    num = diff(yPredict, yArrTest)
    print num
    print '正确率'
    print float(num) / len(yArrTest)

def sigMoidSVM():
    xArr, yArr = loadData('data.txt')
    xArr = regularize(xArr).tolist()
    xArrTrain = xArr[0:2500][:]
    yArrTrain = yArr[0:2500]
    xArrTest = xArr[2500:3000]
    yArrTest = yArr[2500:3000]
    linear_svc = svm.SVC(kernel='sigmoid')
    linear_svc.fit(xArrTrain, yArrTrain)
    yPredict = linear_svc.predict(xArrTest)
    #print yPredict
    print '正确的个数:'
    num = diff(yPredict, yArrTest)
    print num
    print '正确率'
    print float(num) / len(yArrTest)

if __name__ == '__main__':
    print 'rbf:'
    rbfSVM()
    print 'sigmoid:'
    sigMoidSVM()
    print 'linear:'
    linearSVM()
