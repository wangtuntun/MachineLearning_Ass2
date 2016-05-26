#coding=utf-8
'''
Created on 2016年5月13日

@author: pangb
'''
from sklearn.cluster import KMeans
from numpy import *
import math

def regularize(xMat):
    #inMat = xMat.copy()
    inMeans = mean(xMat, 0)
    inVar = var(xMat, 0)
    xMat = (xMat - inMeans) / inVar

    return xMat
class RBFNN:

    # 从文件中读取数据
    @staticmethod
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

    # 计算两个向量之间的距离
    @staticmethod
    def distCal(vecA, vecB):
        return sqrt(sum(power(vecA - vecB, 2)))

    # 为给定数据集构建一个包含k个随机质心的集合
    @staticmethod
    def randCentral(dataSet, k):
        n = shape(dataSet)[1]
        cents = mat(zeros((k, n)))
        for i in range(n):
            minJ = min(dataSet[:, i])
            rangeJ = float(max(dataSet[:, i]) - minJ)
            cents[:, i] = mat(minJ + rangeJ * random.rand(k, 1))
        return cents

    # 求k均值聚类
    @staticmethod
    def kMeans(self, dataSet, k):
        m = shape(dataSet)[0]
        clusterAssMent = mat(zeros((m, 2)))
        cents = self.randCentral(dataSet, k)
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = inf; minIndex = -1
                for j in range(k):
                    distJI = self.distCal(cents[j, :], dataSet[i, :])
                    if distJI < minDist:
                        minDist = distJI; minIndex = j
                if clusterAssMent[i, 0] != minIndex:clusterChanged = True
                clusterAssMent[i, :] = minIndex, minDist ** 2
            for cent in range(k):
                ptsInClust = dataSet[nonzero(clusterAssMent[:, 0].A == cent)[0]]
                cents[cent, :] = mean(ptsInClust, axis=0)
        return cents, clusterAssMent
        # 初始化
        # @i,j,k分别为rbf网络的输入层，隐含层，输出层的节点个数
    def __init__(self, i, j, k):
        self.input_num = i
        self.hide_num = j
        self.output_num = k

        self.weight = ones((self.hide_num, self.output_num))

    # 高斯函数
    @staticmethod
    def gaussian(self, x, y, fai):
        r = sum(power(x - y, 2))
        return exp(-r / (2 *fai))

    # 矩阵的逆求解theta

    @staticmethod
    def standRegres(xArr, yArr):
        xMat = mat(xArr)
        yMat = mat(yArr).T
        xTx = xMat.T * xMat
        if linalg.det(xTx) == 0.0:
            print "该矩阵没有逆矩阵，无法求解"
            return
        ws = xTx.I * (xMat.T * yMat)
        return ws
    
    def lingSolution(self, xArr, yArr, lam):
        xMat = mat(xArr)
        yMat = mat(yArr).T
        xTx  = xMat.T*xMat
        xTemp = xTx + eye(shape(xMat)[1]) * lam
        if linalg.det(xTemp) == 0.0:
            print "该矩阵没有逆矩阵，无法求解"
            return
        ws = xTemp.I * (xMat.T*yMat)
        return ws
    
    # 求中心点之间的最大距离
    def calculateRange(self, cents):
        row = len(cents)
        maxrange = []
        for i in range(row):
            nowRow = cents[i, :]
            tempMat = tile(nowRow, (row, 1))
            diff = (tempMat - cents)
            diff = square(diff)
            
            diffsum = diff.sum(axis=1)
            diffsum = diffsum.tolist()
            diffsum.sort()
            # if(maxrange < sortedDiff[row - 1]):
                # maxrange = sortedDiff[row - 1]
                
            maxrange.append(diffsum[1])

        return maxrange
        
         
    def regularize(self, xMat):
    #inMat = xMat.copy()
        inMeans = mean(xMat, 0)
        inVar = var(xMat, 0)
    
    
        xMat = (xMat - inMeans) / inVar

        return xMat
    
    def rand(self, a, b):
        return (b - a) * random.random() + a
    def calculateWeight(self, basis, yArr, IntegerNum = 1000, rate = 0.01):
        n = len(basis)
        weight = [1.0] * (self.hide_num+1)
        for i in range(self.hide_num):
            weight[i] = self.rand(-1.0, 1.0)
        
        for j in range(IntegerNum):
            for k in range(n):
                yPre = weight * basis[k,:]
                fY = 1 / (1 + exp(-yPre))
                df = fY * (1 - fY)
                err = yArr[k] - df
                weight = weight + rate * err * df * basis[k,:].T
        
        return weight

    def calculateDiff(self, y1, y2):
        count = 0
        for i in range(len(y2)):
            if y1[0][i] > 0.0 and y2[i] > 0.0:
                count += 1
            if y1[0][i] < 0.0 and y2[i] < 0.0:
                count += 1
        return count
    
    def isnan(self, cents):
        tempCents = cents.tolist()
        for i in range(len(tempCents)):
            if isnan(tempCents[i][0]) or isnan(tempCents[i][1]):
                return True
        
        return False
    # 交叉校验k的值
    def rbfCross(self, xArr, yArr, k=5):
        row = len(yArr)
        foldSize = int(math.floor(row / k))
        xArr = self.regularize(mat(xArr)).tolist()
        num1 = 0
        error = 0
        # 交叉校验k次
        for i in range(k):
            xTrain = []
            yTrain = []
            xTest = []
            yTest = []
            # 交叉校验分组
            for j in range(k):
                for m in range(foldSize):
                    if j == i:
                        if(j * foldSize + m >= row):
                            break
                        xTest.append(xArr[j * foldSize + m])
                        yTest.append(yArr[j * foldSize + m])
                    else:
                        if(j * foldSize + m >= row):
                            break
                        xTrain.append(xArr[j * foldSize + m])
                        yTrain.append(yArr[j * foldSize + m])

            # k聚类xTest集合
            clf = KMeans(n_clusters=self.hide_num)
            s = clf.fit(xTest)
            cents = clf.cluster_centers_
            #cents ,clus=  self.kMeans(self, mat(xTest), self.hide_num)
            if self.isnan(cents):
                continue
            
            num1 += 1
            maxRange = self.calculateRange(cents)
            # 得到权重
            basis = ones((len(xTrain), len(cents)+1))
            # for n in range(len(xTrain)):
            for n in range(len(xTrain)):
                for m in range(len(cents)):
                    basis[n, m+1]=self.gaussian(self, xTrain[n], cents[m], maxRange[m])

            
            ws=self.lingSolution(basis, yTrain, 0.001)
            
            #ws = self.calculateWeight(basis, yTrain)
            # 交叉校验误差
            vBasis=ones((len(xTest), len(cents)+1))
            for n in range(len(xTest)):
                for m in range(len(cents)):
                    vBasis[n, m+1]=self.gaussian(self, xTest[n], cents[m], maxRange[m])

            yPredict=vBasis * ws
            #ws = mat(ws)
            #shape1 = shape(ws)
            #yPredict = vBasis * ws.T
            #fY = 1 / (1 + exp(-yPredict))
            yPredict = yPredict.T
            print self.calculateDiff(yPredict.tolist(), yTest)
            matYtest = mat(yTest)
            diff=(yPredict - matYtest)
            diff = square(diff)
            errlist=diff.sum()
            error += errlist
        return error/num1
    
    #测试集合
    def rbfPredict(self, xArr, yArr, xTest):
        row = len(xArr)
            # k聚类xTest集合
        clf = KMeans(n_clusters=self.hide_num)
        s = clf.fit(xArr)
        cents = clf.cluster_centers_
            #cents ,clus=  self.kMeans(self, mat(xTest), self.hide_num)
            
        maxRange = self.calculateRange(cents)
            # 得到权重
        basis = ones((len(xArr), len(cents)+1))
            # for n in range(len(xTrain)):
        for n in range(len(xArr)):
            for m in range(len(cents)):
                basis[n, m+1]=self.gaussian(self, xArr[n], cents[m], maxRange[m])

        #岭回归求得权重 
        ws=self.lingSolution(basis, yArr, 0.001)
            
            #ws = self.calculateWeight(basis, yTrain)
            # 交叉校验误差
        vBasis=ones((len(xTest), len(cents)+1))
        for n in range(len(xTest)):
            for m in range(len(cents)):
                vBasis[n, m+1]=self.gaussian(self, xTest[n], cents[m], maxRange[m])

        yPredict=vBasis * ws
        return yPredict.T
    
def TestCross():

    xArr, yArr= RBFNN.loadData('data.txt')
    xArr = regularize(mat(xArr)).tolist()
    n=RBFNN(2, 10, 1)
    print n.rbfCross(xArr[0:2500][:], yArr[0:2500])

if __name__ == '__main__':
    TestCross()
