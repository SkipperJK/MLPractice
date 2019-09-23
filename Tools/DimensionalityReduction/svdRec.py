import numpy as np


"""numpy
np.array([1,2,4]) 这种默认为是列向量 shape = (3,)

"""

"""numpy linalg
linalg.norm(): Vector or Matrix norm (向量或者矩阵的范数（2范数））


linalg.svd(): Singular value decomposition of matrix 矩阵的奇异值分解
"""

# ？？？ 有个问题，问什么numpy中好多函数返回的结果多个元素的元组，例如:np.nonzero()

def loadExData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]


def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def euclidSim(inA, inB):
    return 1.0/(1.0+np.linalg.norm(inA-inB))


def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    else:
        return 0.5 + 0.5*np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA)*np.linalg.norm(inB)
    # print(num, denom)
    return 0.5 + 0.5*(num/denom)


def standEst(dataMat, user, simMeas, item):
    """
    基于物品相似度的评分估计
    对于user未评级的一个物品itme，通过遍历所有物品，计算该item与所有物品中该user评级过的物品的相似度，物品相似度计算使用协同过滤（列向量相似度）

    :param dataMat: 数据矩阵（假设行为用户，列为物品）
    :param user: 用户编号
    :param simMeas: 相似度计算方法
    :param item: 物品编号（该用户为评级的物品）
    :return:
    """
    n = np.shape(dataMat)[1] # 物品的数量
    simTotal = 0.0
    ratSimTotal = 0.0

    # 该循环大体上就是对用户评过分的每个物品进行遍历，并将它和其他物品进行比较
    for j in range(n):
        # 只针对用户评级过的物品
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A > 0, dataMat[:,j].A > 0))[0] # 寻找所有同时评级了item和j物品的所有用户，得到该用户的下标，用于计算物品相似度
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print("overlap", overLap, 'the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating # 评级的物品与item的相似度 和 当前评分 的乘积
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal # 使得评分归一化，评分值在0-5之间


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    """
    对user所有没有评级的物品，对每个物品计算与该用户评级过的物品相似度，对相似度值进行排序
    :param dataMat:
    :param user:
    :param N: 前N个推荐结果
    :param simMeas:
    :param estMethod:
    :return:
    """
    unratedItems = np.nonzero(dataMat[user, :].A==0)[1] #??? 返回的是ndarray
    if len(unratedItems) == 0:
        return "you rated everything"
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj:jj[1], reverse=True)[:N]


def svdEst(dataMat, user, simMeas, item):
    """
    基于SVD的评分估计
    :param dataMat:
    :param user:
    :param simMeas:
    :param item:
    :return:
    """
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(4)*Sigma[:4])  # 建立对角矩阵
    xformedItems = dataMat.T * U[:, :4] * Sig4.I    # 构建转换后的物品, 转换为低维空间
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item:
            continue
        else:
            similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
            print('the %d and %d similarity is: %f' % (item, j, similarity))
            simTotal += similarity
            ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal


def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1, end='')
            else:
                print(0, end='')
        print(' ')


def imgCompress(numSV=3, thresh=0.8):
    myl = []
    with open('../data/SVD/0_5.txt', 'r') as fr:
        for line in fr.readlines():
            newRow= []
            for i in range(32):
                newRow.append(int(line[i]))
            myl.append(newRow)
    myMat =  np.mat(myl)
    print("****Original matrix******")
    printMat(myMat, thresh)
    U, Sigma, VT = np.linalg.svd(myMat)
    # SigRecon = np.mat(np.zeros((numSV, numSV)))
    # for k in range(numSV):
    #     SigRecon[k,k] = Sigma[k]
    SigRecon = np.eye(numSV)*Sigma[:numSV]
    print(SigRecon)
    reconMat = U[:, :numSV]*SigRecon*VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)




if __name__ == '__main__':
    Data = loadExData()
    U, Sigma, VT = np.linalg.svd(Data)
    print(Sigma)

    # 重构原始矩阵的近似矩阵
    # 结果发现通过奇异值重建的矩阵和原始矩阵很相似
    Sig3 = np.mat([[Sigma[0],0,0],
                   [0,Sigma[1],0],
                   [0,0,Sigma[2]]])

    reconData = U[:,:3]*Sig3*VT[:3,:]
    print(np.shape(reconData))
    print(reconData)

    myMat = np.mat(loadExData())
    print("欧式距离：", euclidSim(myMat[:,0], myMat[:,4]))
    print("欧式距离：", euclidSim(myMat[:,0], myMat[:,0]))
    print("余弦相似度：", cosSim(myMat[:,0], myMat[:,4]))
    print("余弦相似度：", cosSim(myMat[:,0], myMat[:,0]))
    print("皮尔逊相关系数：", pearsSim(myMat[:,0], myMat[:,4]))
    print("皮尔逊相关系数：", pearsSim(myMat[:,0], myMat[:,0]))

    # 对矩阵稍加修改
    myMat[4,1]=myMat[4,0]=myMat[5,0]=myMat[6,0] = 4
    myMat[0,3] = 2

    print(myMat[6,:])
    recdItems = recommend(myMat, 6)
    print(recdItems)
    print(myMat[2,:])
    recdItems = recommend(myMat, 2)
    print(recdItems)

    # 接近真实情况的数据集
    print("\n\n 另一个数据集")
    U, Sigma, VT = np.linalg.svd(np.mat(loadExData2()))
    print(Sigma)
    Sig2 = Sigma**2 # 对每个元素取平方
    print("奇异值所有能量：", sum(Sig2))
    print("奇异值90%能量：", sum(Sig2)*0.9)
    print("奇异值前2个能量：", sum(Sig2[:2]))
    print("奇异值前3个能量：", sum(Sig2[:3]))
    # 发现前三个奇异值高于总能量的90%，可以将11维的矩阵转换维一个3维的矩阵
    recdItems = recommend(myMat, 6, estMethod=svdEst)
    print(recdItems)
    recdItems = recommend(myMat, 6, simMeas=pearsSim, estMethod=svdEst)
    print(recdItems)

    imgCompress(2)
    # 使用SVD矩阵分解，实现高效率的存储和传输