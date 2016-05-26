环境:python2.7, numpy, matplotlib, sklearn机器学习库（可用pip install -U scikit-learn 命令快速安装）

代码说明：
1：bpNetwork.py为BP神经网络的主要功能程序
2：executeBpNetwork.py 是执行各种BPNetwork动作的程序，具体使用如下：
该程序接收一些参数，有-rate, -num, -predict, -norm, -nonorm(建议一次只输入一个参数)
-rate 用来调学习速率的函数，可以画出不同学习速率情况下预测的情况
-num 用来调隐含层的节点个数的函数，可以画出不同节点个数情况下的预测情况
-norm 对x进行z-score规范化后的效果
-nonorm 对x不进行规范化的效果
-predict 根据训练的模型来预测测试集的数据
比如如果想要预测测试集的数据，所需的命令如下：python executeBpNetwork.py -predict
3：rbfNetwork.py为rbf神经网络的主要功能程序
4：executeRbfNetwork.py 是执行rbfNetwrok动作的程序，具体使用如下：
该程序接收一些参数，有-crossNormal, -crossNoNormal, -drawKmeans, -predict(建议一次只输入一个参数)
-crossNormal 用来交叉校验kMeans的k的大小，其中对x进行了z-score规范化
-crossNoNormal 用来交叉校验kMeans的k的大小，其中没有对x进行规范化处理
-drawKmeans 用来画出将x聚成12类的效果
-predict 根据训练的模型来预测测试集的数据
比如如果想要画出将x聚成12类的效果，可使用如下命令 python executeRbfNetwork.py -drawKmeans
5:svm.py 是使用了sklearn的svm库来分别执行以线性，sigmoid和rbf为核函数的svm
