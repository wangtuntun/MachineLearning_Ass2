#coding=utf-8
'''
Created on 2016��5��17��

@author: pangb
'''
from bpNetwork import *
import matplotlib.pyplot as plt
import sys
def testBestLearningRate():
    rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    errNum = []
    for i in range(len(rate)):
        print 'test' + str(i)
        xArr, yArr = loadData('data.txt')
        n = NN(2, 7, 1)
    # 用一些模式训练它
        n.train(xArr[500:2999][:], yArr[500:2999], 500, rate[i])
        errNum.append(n.test(xArr[500:2999][:], yArr[500:2999]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = range(7)
    ax.plot(x1, errNum)
    plt.show()

def testBestNumOfNeural():
    nums = [3,4,5,6,7,8,9,10,11,12]
    errNum = []
    for i in range(len(nums)):
         print 'test' + str(i)
         xArr, yArr = loadData('data.txt')
         n = NN(2, nums[i], 1)
         n.train(xArr[500:2999][:], yArr[500:2999],500)
         errNum.append(n.test(xArr[500:2999][:], yArr[500:2999]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(nums, errNum)
    plt.show()

def testNorm():    
    xArr, yArr = loadData('data.txt')
    # print xArr
    # print xArr
    xArr = regularize(xArr).tolist()
    n = NN(2, 7, 1)
    # 用一些模式训练它
    n.train(xArr[500:2999][:], yArr[500:2999], 500, 0.01)
    print n.test(xArr[500:2999][:], yArr[500:2999])

def testNoNorm():
    xArr, yArr = loadData('data.txt')
    n = NN(2, 7, 1)
    # 用一些模式训练它
    n.train(xArr[500:2999][:], yArr[500:2999], 500, 0.01)
    print n.test(xArr[500:2999][:], yArr[500:2999])

def testPredict():
    xArr, yArr = loadData('data.txt')
    n = NN(2, 7, 1)
    # 用一些模式训练它
    n.train(xArr[500:2999][:], yArr[500:2999], 500, 0.01)
    count, predict = n.testP(xArr[0:500][:], yArr[0:500])
    print count
    x1 = range(500)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, predict)
    plt.show()
    
if __name__ == '__main__':
	for arg in sys.argv:
		if arg == '-rate':
			testBestLearningRate()
		if arg == '-num':
			testBestNumOfNeural()
		if arg == '-predict':
			testPredict()
		if arg == '-norm':
			testNorm()
		if arg == '-nonorm':
			testNoNorm()
    #testPredict()
    