import numpy as np



'''
CART (Classification And Regression Trees) 分类回归树
决策树不断将数据切分成小数据集，直到所有目标变量完全相同，或者数据不能再切分为止。决策树是一种贪心算法，它要
再给定时间内做出最佳选择，单并不关心能否达到全局最优。
决策树相比于其他机器学习算法的优势之一在于结果更易理解。

树回归方法：模型树，回归树
一般的回归方法
模型树的可解释性优于回归树，也具有更高的预测准确度。

'''


'''python
map()  映射函数
高阶函数map,filter,zip
    python2.x 中返回的是list
    Python3.x 中返回类似迭代器的对象
'''

'''numpy
np.nonzero(): Return the indices of the element that are non-zero

array 和 matrix 的区别
x = np.array([1,2])
model = np.array([[3],[4]])
x*model 得到一个2*2的矩阵

x = np.mat([1,2])
model = np.mat([[3],[4]])
x*model 得到一个值
'''

'''matplotlib
ax.scatter(x,y) 绘制散点图
'''




class treeNode():
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left


def loadDataSet(fileName):
    dataMat = []
    with open(fileName, 'r') as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine)) # 将每行映射为浮点数
            dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    """
    根据指定的feature以及该特征的切分点值切分数据
    :param dataSet:
    :param feature: 待切分特征
    :param value: 待切分特征的某个值
    :return:
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]  # nonzero返回的是非零的元素的下标，然后再取指定下标元素的向量
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    """
    负责生成叶节点，在回归树中，该模型就是目标变量的均值
    :param dataSet:
    :return:
    """
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """
    返回数据集的平方误差的总值
    :param dataSet:
    :return:
    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0] # var求得的是数据集平方误差的均值（均方差），这里需要的是平方误差的总值（总方差）


def chooseBestSplit(dataSet, leafType, errType, ops):
    """
    用最佳的方式切分数据集和生成相应的叶节点，有三种情况不会切分，而是直接创建叶节点，如果找到了一个"好"的切分方式，返回特征编号和切分特征值
    :param dataSet:
    :param leafType:
    :param errType:
    :param ops:
    :return:
    """
    tolS = ops[0] # 容许的误差下降值
    tolN = ops[1] # 切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: # 如果所有的值都相等则退出
        # 统计不同剩余特征值的数目
        return None, leafType(dataSet)
    m,n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex].A1): # 将matrix对象转换为1-d array
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:  # 如果误差减小不大则退出
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):    # 如果切分出的数据集很小则退出
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """
    树的构建算法，递归
        树的构建算法其实对输入的参数tolS和tolN非常敏感
    :param dataSet:
    :param leafType:  建立叶节点的函数
    :param errType: 误差计算函数
    :param ops: 其他参数元组
    :return:
    """
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops) # 满足停止条件时返回叶节点值
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def plotDataSet(x, y):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x,y) # 绘制散点图
    plt.show()


def isTree(obj):
    return (type(obj).__name__=='dict')


def getMean(tree):
    """
    递归函数，从上而下遍历树知道叶节点为止
        如果找到两个叶节点，则计算它们的平均值。该函数对树进行塌陷处理（即返回树的平均值）
    :param tree:
    :return:
    """
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0: return getMean(tree) # 没有测试数据则对树进行塌陷处理

    # 如果有一棵子树为非叶节点，按照切分点切分测试集，并递归调用
    if (isTree(tree['right'])) or (isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if (isTree(tree['left'])):
        tree['left'] = prune(tree['left'], lSet) # 如果合并，树得到从下往上的修改
    if (isTree(tree['right'])):
        tree['right'] = prune(tree['right'], rSet) # 如果合并，树得到从下往上的修改

    # 如果两棵子树都是叶节点，则比较合并前和合并后的误差大小，决定是否合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 求合并前平方误差的总值
        # errorNoMerge = sum(np.power(lSet[:,-1] - tree['left'],2)) +\
        #                sum(np.power(rSet[:,-1] - tree['right'], 2))
        errorNoMerge = sum(np.power(lSet[:, tree['spInd']] - tree['left'], 2)) + \
                       sum(np.power(rSet[:, tree['spInd']] - tree['right'], 2))
        treeMean = (tree['left']+tree['right'])/2.0 # 合并后的平方误差总值
        # errorMerge = sum(np.power(testData[:,-1] - treeMean, 2))
        errorMerge = sum(np.power(testData[:,tree['spInd']] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def linearSolve(dataSet):
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n)))
    Y = np.mat(np.ones((m,1)))
    # 将X与Y中的数据格式化
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, \n try increasing the second value of ops')
    ws = xTx.I * (X.T * Y) # 标准回归，最小二乘法
    return ws,X,Y


def modelLeaf(dataSet):
    """
    数据不再需要切分的时候负责生成叶节点
    :param dataSet:
    :return:
    """
    ws,X,Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    """
    在给定的数据集上计算误差，yHat和y的平方误差
    :param dataSet:
    :return:
    """
    ws,X,Y = linearSolve(dataSet)
    yHat = X*ws
    return sum(np.power(Y - yHat, 2))


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    """
    模型树，叶节点为模型，即回归参数的列向量
    :param model:
    :param inDat:
    :return:
    """
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1] = inDat # 在原数据矩阵上增加第0列
    return float(X*model) # 回归模型


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat




if __name__ == '__main__':

    # Test binSplitDataSet
    # testMat = np.mat(np.eye(4))
    # mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    # print(mat0)
    # print(mat1)


    # 数据集 ex00.txt
    # myDat = loadDataSet('../data/TreeRegression/ex00.txt')
    # plotDataSet(np.array(myDat)[:,0], np.array(myDat)[:,1])
    # myMat = np.mat(myDat)
    # tree = createTree(myMat)
    # print(tree)


    # 数据集 ex0.txt 该数据集样本向量为三维，第0维全部相等为1
    # myDat1 = loadDataSet('../data/TreeRegression/ex0.txt')
    # plotDataSet(np.array(myDat1)[:,1], np.array(myDat1)[:,2])
    # myMat1 = np.mat(myDat1)
    # tree = createTree(myMat1)
    # print(tree)


    # 数据集 ex2.txt
    # myDat2 = loadDataSet('../data/TreeRegression/ex2.txt')
    # myArr2 = np.array(myDat2)
    # plotDataSet(myArr2[:,0], myArr2[:,1])
    # myMat2 = np.mat(myDat2)
    # tree = createTree(myMat2)
    # print(tree)
    # # 测试数据集 ex2test.txt
    # myDatTest = loadDataSet('../data/TreeRegression/ex2test.txt')
    # myArrTest = np.array(myDatTest)
    # myMatTest = np.mat(myDatTest)
    # pTree = prune(tree, myMatTest)
    # print(tree)


    # 数据集 exp2.txt
    # 模型树，每个节点是一个线性模型，表示为模型的系数
    # myDatM2 = loadDataSet('../data/TreeRegression/exp2.txt')
    # myArrM2 = np.array(myDatM2)
    # myMatM2 = np.mat(myDatM2)
    # plotDataSet(myArrM2[:,0], myArrM2[:,1])
    # tree = createTree(myMatM2, modelLeaf, modelErr)
    # print(tree)


    # 数据集 bikeSpeedVsIq_test.txt, bikeSpeedVsIq_train.txt
    trainDat = loadDataSet('../data/TreeRegression/bikeSpeedVsIq_train.txt')
    testDat = loadDataSet('../data/TreeRegression/bikeSpeedVsIq_test.txt')
    trainArr = np.array(trainDat)
    testArr = np.array(testDat)
    trainMat = np.mat(trainDat)
    testMat = np.mat(testDat)
    plotDataSet(trainArr[:,0], trainArr[:,1])
    plotDataSet(testArr[:,0], testArr[:,1])
    # 创建回归树
    myTree = createTree(trainMat)
    print(myTree)
    yHat = createForeCast(myTree, testMat[:,0])
    corrcoef = np.corrcoef(yHat, testMat[:,1], rowvar=0)[0,1]
    print(corrcoef)
    # 创建模型树
    myTree = createTree(trainMat, modelLeaf, modelErr, (1,20))
    print(myTree)
    yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
    corrcoef = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0,1]
    print(corrcoef)
        # 预测值与真实值的相关系数, R^2, 越接近1越好
    # 标准的线性回归
    ws,X,Y = linearSolve(trainMat)
    print(ws)
    for i in range(np.shape(testMat)[0]):
        yHat[i] = testMat[i,0]*ws[1,0]+ws[0,0]
    corrcoef = np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
    print(corrcoef)

