import numpy as np
import math

"""
PCA的降维，其本质是将高维空间的向量投影到一个低纬空间里。其关键就是找到最合适的投影空间。让原来的高维矩阵投影到这个平面以后能尽可能多得保留原有的信息。
"""

"""numpy linalg（线性代数工具箱）
Core Linear Algebra Tools
    det             Determinant of a square matrix

    eig             Eigenvalues and vectors of a square matrix
    
    
mat[:, [0,1,4]] # 取mat中指定的列
np.mean()  Return ndarray
nonzero(a):  Return the indices of the elements that are non-zero

"""

"""
a = float('nan')  # 表示Not a Number 变量 
b = float('inf') # 表示无穷大变量

math.isnan(a) # 判断一个变量是否是NaN
"""


def loadDataSet(fileName, delim='\t'):
    with open(fileName, 'r') as fr:
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        datArr = [list(map(float, line)) for line in stringArr]
        return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    print("meanVals shape: ", np.shape(meanVals))
    meanRemoved = dataMat - meanVals # 去除平均值
    print("meanRemoved shape: ", np.shape(meanRemoved))
    covMat = np.cov(meanRemoved, rowvar=0)  # 计算协方差矩阵，变为方阵 square matrix
    print("cov matrix shape: ", np.shape(covMat))
    eigVals, eigVects = np.linalg.eig(np.mat(covMat)) # 计算协方差矩阵的特征值和特征向量
    eigValInd = np.argsort(eigVals) # 对特征值进行从小到大的排序   ？？？ 为什么不直接使用np.argsort(-eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1] # 将排序结果逆序  # [索引:索引:步长]
    redEigVects = eigVects[:, eigValInd] # 获得指定特征值对应的特征向量（这里的eigVects，其中的列向量为特征向量）
    print("redEigVects shape: ", np.shape(redEigVects))

    lowDDataMat = meanRemoved * redEigVects # 将数据转换到新空间 降维之后的矩阵
    print("low Dimensionality matrix shape: ", np.shape(lowDDataMat))
    reconMat = (lowDDataMat * redEigVects.T) + meanVals # 重构之后的数据
    print('reconMat shape: ', np.shape(reconMat))
    return lowDDataMat, reconMat


def plotLowDDat(dataMat, reconMat):
    """
    绘制原始数据和重构之后的数据
    :param dataMat:
    :param reconMat:
    :return:
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()


def replaceNanWithMean():
    datMat = loadDataSet('../data/PCA/secom.data', ' ')
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~math.isnan(datMat[:,i].A))[0], i]) # 对第i维的特征求均值
        datMat[np.nonzero(math.isnan(datMat[:,i].A))[0], i] = meanVal
    return datMat

if __name__ == '__main__':
    dataMat = loadDataSet('../data/PCA/testSet.txt')
    print("DataSet shape: ", np.shape(dataMat))
    lowDMat, reconMat = pca(dataMat, 1)
    plotLowDDat(dataMat, reconMat)


    # 数据集：secom.data  利用PCA对半导体制造数据降维
    dataMat = replaceNanWithMean()
