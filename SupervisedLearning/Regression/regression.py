import numpy as np

"""
线性回归
    线性回归的一个问题是有可能出现欠拟合的现象，因为它求得是最小均方误差的**无偏估计**。
    
局部加权线性回归(Locally Weighted Linear Regression LWLR)
    在估计中引入一些偏差，从而降低预测的均方误差
    但是增加了计算量，因为它对每个点做预测时都必须使用整个数据集，必须保存整个数据集
    
岭回归
    缩减系数来"理解"数据，
    通过观察在缩减过程中回归系数是如何变化的，从而挖掘数据的内在规律

前向逐步回归
    缩减系数来"理解"数据
    主要的优点可以帮助人们理解现有的模型并做出改进，运行该算法找出重要的特征，就有可能及时停止对哪些不重要的特征的收集
    
当应用缩减方法（如逐步线性回归和岭回归）时，模型也就增加了偏差（bias），与此同时却减少了模型的方差。
    
"""

"""numpy
matrix.argsort()
np.argsort(matrix)
matrix.A  base array 将矩阵转换为数组(按照matrix原始的结构）
matrix.A1 1-d base array 将matrix转换为一个维数组

"""

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    with open(fileName, 'r') as fr:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    """
    最小二乘法，对平方误差求最小值，求解对应的回归系数
    :param xArr:
    :param yArr:
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0: # 计算行列式是否为0
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


def plotLine(xArr, yArr, ws):
    import matplotlib.pyplot as plt
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制训练数据点
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0]) # 取样本特征向量的第一维作为x轴
    # 绘制最佳拟合曲线
    yHat = xMat*ws
    xCopy = xMat.copy()
    xCopy.sort(0)   # 如果直线上的数据点次序混乱，绘图时将会出现问题
    yHat = xCopy*ws
    ax.plot(xCopy[:,1], yHat)
    plt.show()


def calCorrcoef(yHat, yMat):
    """
    计算两个序列的相关系数，得到一个2*2的相关矩阵，包含所有两两组合，对角线上的数据是1，因为序列和自己匹配是最完美的。
    :param yHat:
    :param yMat:
    :return:
    """
    result = np.corrcoef(yHat.T, yMat)
    print(result)


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m))) # 创建对角矩阵，为每个样本点初始化了一个权重
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))    # 权重的大小以指数级衰减
        # 计算每个样本点的权重：随着样本点与待预测点距离的递增，权重将以指数级衰减
    xTx = xMat.T * (weights * xMat) # weight是一个对角矩阵，对角线上的每个元素为对应样本的权重
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def plotLwlr(xMat, yMat, yHat):
    import matplotlib.pyplot as plt
    srtInd = xMat[:, 1].argsort(0) # 按行排序
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s=2, c='red')
    plt.show()


def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum() # 误差平方和


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 数据标准化：所有的特征减去各自的均值并除以方差
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)  # 方差
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    # 在30个不同的lambda下调用ridgeRegres()函数
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i,:] = ws.T
    return wMat


def plotRidge(ridgeWeights):
    """
    岭回归的回归系数变化图
    :param ridgeWeights:
    :return:
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


def regularize(xMat):
    """
    0均值标准化：即均值为0方差为1
    :param xMat:
    :return:
    """
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMean)/xVar
    return xMat


def stageWise(xArr, yArr, eps=0.01, numIt=100):
    """

    :param xArr:
    :param yArr:
    :param eps: 每次迭代需要调整的步长
    :param numIt: 迭代的次数
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 数据标准化
    # 0均值
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    # 0均值标准化
    xMat = regularize(xMat)
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = float('inf')
        # 贪心算法在所有特征上运行两次for循环，分别计算增加和减少该特征对误差的影响
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign   # 增加和减少某个特征对误差的影响
                yTest = xMat*wsTest
                rssE = rssError(yMat.A, yTest.A) # matrix.A 代表将矩阵转换为数组
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat


if __name__ == '__main__':
    xArr, yArr = loadDataSet("../data/LinearRegression/ex0.txt")
    # print(len(xArr))
    xMat = np.mat(xArr)
    print(np.shape(xMat))
    yMat = np.mat(yArr).T

    # 线性回归
    # ws = standRegres(xArr, yArr)
    # print(ws)
    # plotLine(xArr, yArr, ws)
    # yHat = xMat * ws
    # print(np.shape(yHat), np.shape(yMat))
    # calCorrcoef(yHat, yMat.T)

    # 局部加权线性回归
    # print(lwlr(xArr[0], xArr, yArr, 1.0))
    # print(lwlr(xArr[0], xArr, yArr, 0.001))
    # yHat = lwlrTest(xArr, xArr, yArr, 1.0)
    # plotLwlr(xMat, yMat, yHat)
    # yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    # plotLwlr(xMat, yMat, yHat)
    # yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    # plotLwlr(xMat, yMat, yHat)
    '''
    如图：
        k=1.0时的模型效果与最小二乘法差不多
        k=0.01时该模型可以挖出数据的潜在规律
        k=0.003时则考虑了太多的噪声，进而导致了过拟合现象
    '''

    # example 1
    abX, abY = loadDataSet("../data/LinearRegression/abalone.txt")
    # yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    # yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    # yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    # print(rssError(abY[0:99], yHat01.T))
    # print(rssError(abY[0:99], yHat1.T))
    # print(rssError(abY[0:99], yHat10.T))
    # yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    # yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    # yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    # print(rssError(abY[100:199], yHat01.T))
    # print(rssError(abY[100:199], yHat1.T))
    # print(rssError(abY[100:199], yHat10.T))
    '''
    通过上面两个测试得，lwlr必须在未知数据上比较效果才能选取到最佳模型
    '''

    # 岭回归
    # ridgeWeights = ridgeTest(abX, abY)
    # # print(ridgeWeights)
    # plotRidge(ridgeWeights)
    '''
    lambda非常小时，系统与普通回归一样。而lambda非常大时，所有回归系数都为0。
    可以在中间某处找到使得预测的结果最好的lambda值
    '''

    # 前向逐步回归
    xArr, yArr = loadDataSet("../data/LinearRegression/abalone.txt")
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    stageWeights = stageWise(xArr, yArr, 0.01, 200)
    # 这个数据集值得注意的是w1和w6都是0，表明它们不对目标值造成任何影响，也就是说这些特征很可能不需要
    stageWeights = stageWise(xArr, yArr, 0.001, 5000)
    weights = standRegres(xArr, yArr)
    print(weights)
    # 可以看到在5000次迭代后，逐步线性回归算法与常规的最小二乘法效果类似
