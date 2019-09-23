import numpy as np
from SupervisedLearning.Classification.AdaBoost.boost import buildStump, stumpClassify

'''例子
利用多个单层决策树和AdaBoost预测患有马疝病的马是否能够存活

随着弱分类器数目的增加，发现测试错误率在达到一个最小值之后有开始上升了，这类现象称之为过拟合（overfitting）


很多人认为，AdaBoost和SVM是监督机器学习中最强大的两种方法。实际上两者有不少相似之处，可以把弱分类器想象成SVM中的一个核函数，
也可以按照最大化某个最小间隔的方式重写AdaBoost算法，而他们的不同之处就在于其定义的间隔计算方式有所不同，因此导致结果不同。特别是在高维空间下，两者差异更明显。
'''

def loadSimpData():
    '''
    一个简单的数据集
    :return:
    '''
    datMat = np.matrix([[1., 2.1],
                        [2., 1.1],
                        [1.3, 1.],
                        [1., 1.],
                        [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    经过训练得到一个单层决策树的数组, DS代表decision stump单层决策树
    :param dataArr:
    :param classLabels:
    :param numIt:
    :return:
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m) # 初始权重相等
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt): # 迭代numIt次
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D: ", D.T)
        # 计算当前分类器的权重（错误率越高，分类器的权重越低）
        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16))) # 确保不会发生除零溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T) # 最佳单棵决策树对应的预测值
        # 更新特征权重
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst) # ???
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst # 记录每个数据点的类别估计累计值
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1))) # 由于是类别的加权值，使用sign()函数二值分类的到样本的类别
        errorRate = aggErrors.sum()/m
        print("total error: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    '''
    利用训练出来的多个弱分类器进行分类, 分类结果会随着迭代的进行而越来越强
    :param datToClass: 一个或者多个待分类样例
    :param classifierArr: 多个弱分类器组成的数组
    :return:
    '''
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


def loadDataSet(fileName):
    """
    训练样本的标签要为 1 或者 -1
    :param fileName:
    :return:
    """
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    with open(fileName, 'r') as fr:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat-1):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            if int(float(curLine[-1])) == 0:
                labelMat.append(float(-1))
            else:
                labelMat.append(float(1))
            # labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def plotROC(predStrengths, classLabels):
    """
    ROC曲线给出的是当前阈值变化时假阳率和真阳率的变化情况
        x轴为假阳率：即预测为正实际为负的样本占所有负样本的比例
        y轴为真阳率：即召回率，预测为正实际为正的样本占所有正样本的比例
        左下角的点所对应的是将所有样例判为反例，右上角是将所有样例判为正例的情况。虚线给出的是随机猜测的结果曲线。
    :param predStrengths:
    :param classLabels:
    :return:
    """
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLabels)==1.0) # 训练样本中为正例的个数
    # x轴和y轴区间为[0.0, 1.0]
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort() # 升序排序的元素下标，ndarray类型
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # 按照强度，从右上角(1.0, 1.0)开始画
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0,1], [0,1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)


if __name__ == '__main__':
    # datMat, classLabels = loadSimpData()
    # classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
    # # print(classifierArray)
    # adaClassify([0,0], classifierArray)


    datArr, labelArr = loadDataSet("../../data/LR/horseColicTraining.txt")
    classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)
    testArr, testLabelArr = loadDataSet("../../data/LR/horseColicTest.txt")
    prediction10 = adaClassify(testArr, classifierArray)
    errArr = np.mat(np.ones((67,1)))
    print(errArr[prediction10 != np.mat(testLabelArr).T].sum()) # 67个中有16的预测错误，所以测试误差为 16/67
    plotROC(aggClassEst.T, labelArr)

