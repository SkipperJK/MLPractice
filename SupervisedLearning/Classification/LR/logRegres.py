import numpy as np
import matplotlib.pyplot as plt

'''例子
使用Logistic回归来预测患有马疝病的马是否能够存活
'''

"""
逻辑回归中的优化算法：
    批量梯度上升：使用矩阵运算，每次更新遍历整个训练集，迭代多次
    随机梯度上升：每次更新选择一个样本，alpha固定，样本不是随机选取
    改进的随机梯度上升：每次更新随机地选择一个样本，alpha在每次迭代中更新，这里的没遍历一遍所有的训练集算迭代一次
    
随机梯度上升是一个在线算法，可以在新数据到来的时就完成参数更新，不需要重新读取整个数据集来进行批处理运算。
    
这里有30%的数据缺失，如果调整colicTest()中的迭代次数和stocGradAscent1()中的步长，平均错误率可以降到20%。
"""


"""numpy

array:
    array*array: 两个array的shape相同，乘法运算是元素与对应元素相乘，得到的结果shape依旧是array的shape

"""
def loadDataSet():
    dataMat = []
    labelMat = []
    with open("../data/LR/testSet.txt", 'r') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # 1.0是什么意思。哦，是常数项
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    # RuntimeWarning: overflow encountered in exp, 可以将数据格式转换为 dtype=np.float128
    return 1.0 / (1 + np.exp(-inX)) # np.exp() 对np.array / np.matrix中的每个元素执行指数运算


def gradAscent(dataMatIn , classLabels):
    """
    批量梯度上升，每次更新要遍历整个数据集
    这里，更新时，对所有样本使用矩阵运算，而不是一个一个运算后求和
    :param dataMatIn:
    :param classLabels:
    :return:
    """
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()  # 对矩阵进行转置,变为 m*1
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))    # 初始化回归系数为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)   # 矩阵相乘， m*n乘n*1，得到m*1的矩阵
        error = (labelMat - h)  # 真实类别与预测类别的差值
        weights = weights + alpha * dataMatrix.transpose() * error  # 矩阵相乘， n*m乘m*1 得到n*1, alpha * matrix是对矩阵中的每个元素乘alpha
        # 矩阵运算： \theta = \theta + \alpha * x^{T} * E
    return weights


def stocGradAscent(dataMatrix, classLabels):
    """
    随机梯度上升，每次用一个样本做更新
    :param dataMatrix:
    :param classLabels:
    :return:
    """
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights)) # 由于dataMatrix是array类型，array相乘 是元素与对应元素相乘
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进随机梯度上升
        每次迭代调整alpha: 可以缓解迭代过程中 参数 的波动或者高频波动
        随机选取用于更新的样本，（不重复): 可以减少周期性波动
        收敛速度更快
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    """
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0+j+i) + 0.01    # 学习步长 alpha 并不是严格下降的
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            # print(randIndex, len(dataIndex))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')   # 画散点图，x,y可以是list
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)   # start, stop, step  return array 这里的意思是为了画直线，取60 x坐标值
    print(type(weights))
    weights = np.array(weights) # 由于传过来的参数weights为matrix，x.shape=(60,) 如果与matrix运算之后y.shape=(1,60)
    y = (-weights[0]-weights[1]*x)/weights[2]   # 这里 x 是array,因此无论是 加减乘除 都是把array中的每个元素当作运算单位
    print(x.shape, y.shape)
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVector(inX, weights):
    """

    :param inX:
    :param weights: 是array类型
    :return:
    """
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open("../data/LR/horseColicTraining.txt")
    frTest = open("../data/LR/horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1

    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: {}".format(errorRate))
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))




if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # weights = gradAscent(dataArr, labelMat)
    # print(weights)
    # plotBestFit(weights)
    # weights = stocGradAscent(np.array(dataArr), labelMat)
    # plotBestFit(weights)
    # weights = stocGradAscent1(np.array(dataArr), labelMat)
    # plotBestFit(weights)
    multiTest()