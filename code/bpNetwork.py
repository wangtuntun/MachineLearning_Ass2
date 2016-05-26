# coding=utf8
import math
import random
import string
import numpy
import sys

random.seed(0)

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
 # 生成区间[a, b)内的随机数


def rand(a, b):
    return (b - a) * random.random() + a

def sigmoid(x):
    a = 1 / (1 + numpy.exp(-x))
    return a
    #return math.tanh(x)

 # 函数 sigmoid 的派生函数, 为了得到输出 (即：y)


def dsigmoid(y):
    #return 1.0 - y**2
    return y * (1-y)


class NN:
    def __init__(self, ni, nh, no):
        # 输入层、隐藏层、输出层的节点（数）
        self.input_num = ni + 1  # 增加一个偏差节点
        self.hide_num = nh
        self.output_num = no

        # 激活神经网络的所有节点（向量）
        self.input_arr = [1.0] * self.input_num
        self.hide_arr = [1.0] * self.hide_num
        self.output_arr = [1.0] * self.output_num

        # 建立权重（矩阵）
        self.inHide_weight = numpy.ones((self.input_num, self.hide_num)).tolist()
        self.hideOut_weight = numpy.ones((self.hide_num, self.output_num)).tolist()
        # 设为随机值
        for i in range(self.input_num):
            for j in range(self.hide_num):
                self.inHide_weight[i][j] = rand(-1.0, 1.0)
        for j in range(self.hide_num):
            for k in range(self.output_num):
                self.hideOut_weight[j][k] = rand(-1.0, 1.0)
    def update(self, inputs):

        # 激活输入层
        for i in range(self.input_num - 1):
            self.input_arr[i] = inputs[i]

        # 激活隐藏层
        for j in range(self.hide_num):
            sum1 = 0.0
            for i in range(self.input_num):
                sum1 = sum1 + self.input_arr[i] * self.inHide_weight[i][j]
            self.hide_arr[j] = sigmoid(sum1)

        # 激活输出层
        for k in range(self.output_num):
            sum2 = 0.0
            for j in range(self.hide_num):
                sum2 = sum2 + self.hide_arr[j] * self.hideOut_weight[j][k]
            self.output_arr[k] = sigmoid(sum2)

        return self.output_arr[:]

    def backPropagate(self, targets, N):
        ''' 反向传播 '''

        # 计算输出层的误差
        errorlist = [0.0] * self.output_num
        for k in range(self.output_num):
            error = targets[k] - self.output_arr[k]
            errorlist[k] = dsigmoid(self.output_arr[k]) * error

        # 计算隐藏层的误差
        hide_errlist = [0.0] * self.hide_num
        for j in range(self.hide_num):
            error = 0.0
            for k in range(self.output_num):
                error = error + errorlist[k] * self.hideOut_weight[j][k]
            hide_errlist[j] = dsigmoid(self.hide_arr[j]) * error

        # 更新输出层权重
        for j in range(self.hide_num):
            for k in range(self.output_num):
                change = errorlist[k] * self.hide_arr[j]
                self.hideOut_weight[j][k] = self.hideOut_weight[j][k] + N * change
        # 更新输入层权重
        for i in range(self.input_num):
            for j in range(self.hide_num):
                change = hide_errlist[j] * self.input_arr[i]
                self.inHide_weight[i][j] = self.inHide_weight[i][j] + N * change

        # 计算误差
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.output_arr[k])**2
        return error

    def test(self, xArr, yArr):
        count = 0
        for i in range(len(yArr)):
            y1 = self.update(xArr[i])[0]
            y2 = yArr[i]
            print(y2, '->', y1)
           

            if (y1 >= 0.5) and (y2 >= 0.0):
                count += 1
            if (y1 < 0.5) and (y2 < 0.0):
                count += 1
            
            #if abs(y1 - y2) < 1:
             #   count += 1;
                
        return count
    
    #返回预测的y与正确的count值
    def testP(self, xArr, yArr):
        count = 0
        predict = []
        for i in range(len(yArr)):
            y1 = self.update(xArr[i])[0]
            y2 = yArr[i]
            print(y2, '->', y1)
           
            predict.append(y1)
            if (y1 >= 0.5) and (y2 >= 0.0):
                count += 1
            if (y1 < 0.5) and (y2 < 0.0):
                count += 1
            
            #if abs(y1 - y2) < 1:
             #   count += 1;
                
        return count,predict



    def train(self, xArr, yArr, iterations=1000, N=0.01):
        # N: 学习速率(learning rate)
        for i in range(iterations):
            error = 0.0
            for j in range(len(yArr)):
                inputs = xArr[j]

                targets = []
                targets.append(yArr[j])

                self.update(inputs)
                error = error + self.backPropagate(targets, N)
            if i % 100 == 0:
                print('err %-.5f' % error)


def testMain():
    xArr, yArr = loadData('data.txt')
    # print xArr
    # print xArr
    #xArr = regularize(xArr).tolist()
    n = NN(2, 10, 1)
    # 用一些模式训练它
    n.train(xArr[500:2999][:], yArr[500:2999])
    # 测试训练的成果（不要吃惊哦）
    print n.test(xArr[0:499][:], yArr[0:499])
    # 看看训练好的权重（当然可以考虑把训练好的权重持久化）
    # n.weights()


if __name__ == '__main__':
    testMain()
