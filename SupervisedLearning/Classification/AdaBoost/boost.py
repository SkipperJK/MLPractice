import numpy as np
# from Classification.AdaBoost.adaboost import loadSimpData


'''python numpy

    数组过滤
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

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    通过阈值比较对数据进行分类，样本标签为1/-1
    :param dataMatrix:
    :param dimen:
    :param threshVal: 分类阈值
    :param threshIneq: 比较类型
    :return:
    '''
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        # 数组过滤
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0 # 所有样本的指定dimen的值，进行比较, 将ndarray中满足条件的元素置为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    在样本的特征向量上，通过使得加权错误率最小，找到单棵决策树最佳的切分特征以及对应的切分值，得到最佳单层决策树
    如果错误率为0或者达到最大的迭代次数，则结束
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = float('inf') # 无穷大
    for i in range(n):  # 遍历每个特征
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):    # 对每个步长
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0   # 将预测正确的样本置为0
                weightedError = D.T*errArr  # 计算加权错误率
                # print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy() # 最佳单棵决策树对应的预测值
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    print(bestStump, minError, bestClasEst)
    return bestStump, minError, bestClasEst


if __name__ == '__main__':
    D = np.mat(np.ones((5,1))/5)
    datMat, classLabels = loadSimpData()
    buildStump(datMat, classLabels, D)