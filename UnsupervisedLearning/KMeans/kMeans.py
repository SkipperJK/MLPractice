import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""
    在使用kMeans聚类的时候，其中参数k是用户指定的，但是有可能会发生算法收敛但是聚类效果较差的情况，因为可能收敛到了局部最小值，而非全局最小值。
    为克服kMeans算法收敛与局部最小值的问题，提出了另一种称为 二分K-均值（bisecting K-means）算法。该算法首先将所有点作为一个蔟，然后将该蔟一分为二。
        之后选择其中一个蔟继续进行划分，选择哪一个蔟进行划分取决于对其划分是否可以最大程度降低SSE的值，上述基于SSE的划分过程不断重复，直到得到用户指定的蔟数目为止。
            另一种做法就是选择SSE最大的蔟进行划分，直到蔟数目达到用户指定的数目为止。
    
    一种用于度量聚类效果的指标是SSE（Sum of Square Error）误差平方和
    
    
    K-均值算法
    二分K-均值算法
    层次聚类算法
"""


'''numpy
matrix和array的区别

matrix类型
a = matrix([[0.4775338 , 0.6268906 , 0.16058242],
        [0.85788968, 0.40503986, 0.20374457],
        [0.54098303, 0.10932561, 0.41492208],
        [0.22620567, 0.50215774, 0.96703679]])
a[1,:] = matrix([[0.4775338 , 0.6268906 , 0.16058242]])

array类型
a[1,:].A = array([[0.4775338 , 0.6268906 , 0.16058242]])
a[1,:].A1 = array([0.4775338 , 0.6268906 , 0.16058242])


np.random.rand(m,n)
np.random.random((m,n))

'''

def loadDataSet(filename):
    dataMat = []
    with open(filename, 'r') as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))
            dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA-vecB, 2)))


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1) # np.random.rand(m,n) 生成array
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2))) # 初始化每个样本所属蔟，和与中心点的距离
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 重新计算每个样本距离所属蔟
        for i in range(m):
            minDist = float('inf')
            minIndex = -1
            for j in range(k):
                # distJI = distMeas(centroids[j,:], dataSet[i,:])
                distJI = distMeas(centroids[j,:].A1, dataSet[i,:].A1) # 要转换成array类型
                # print(centroids[j,:], dataSet[i,:])
                # print(distJI, minDist)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        # print(centroids)
        # 更新质心位置
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]  # array 和 常数做比较，每个元素变为 True/False
            centroids[cent,:] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def plotKMeans(dataSet, k, centroids, clustAssment):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:,0], dataSet[:,1])
    # for i in range(k):
    ax.scatter(centroids[:,0].A1, centroids[:,1].A1,s=40, c='red')
    plt.show()


def biKMeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0] # 创建一个初始蔟（所有样本均值）
    centList = [centroid0]
    print(centList)
    for j in range(m):
        clusterAssment[j,1] = distMeas(np.array(centroid0), dataSet[j,:].A1)**2
    while(len(centList) < k):
        lowestSSE = float('inf')
        for i in range(len(centList)): # 对于当前所有已有的蔟，计算每个蔟划分之后的SSE值，选择最小的值进行划分，从而最大程度的降低SSE值。
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])  # 计算当前蔟划分之后的 平方误差和
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0], 1]) # 计算剩余数据的误差和
            # 将划分之后的误差与剩余数据集的误差 之和 作为作为本次误差
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            # 如果该划分的SSE值最小，则本次划分被保存
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 更新蔟的分配结果，划分之后多出来一簇
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList) # 新添加的蔟的编号放到最后
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit
        print("the bestCentToSplit is: ", bestCentToSplit)
        print("the len of bestClustAss is: ", len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]   # 转换成list对象
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0], :] = bestClustAss # 拆分之后，被拆分的蔟的ass改变，更新
    print(centList)
    return np.mat(centList), clusterAssment


def distSLC(vecA, vecB):
    """
    球面余弦定理计算两个 经纬度 之间的距离
    :param vecA:
    :param vecB:
    :return:
    """
    # print(vecA.shape)
    # print(vecB.shape)
    # a = np.sin(vecA[0,1]*np.pi/180) * np.sin(vecB[0,1]*np.pi/180)
    # b = np.cos(vecA[0,1]*np.pi/180) * np.cos(vecB[0,1]*np.pi/180) *\
    #     np.cos(np.pi * (vecB[0,0]-vecA[0, 0])/180)
    a = np.sin(vecA[1] * np.pi / 180) * np.sin(vecB[1] * np.pi / 180)
    b = np.cos(vecA[1] * np.pi / 180) * np.cos(vecB[1] * np.pi / 180) * \
        np.cos(np.pi * (vecB[0] - vecA[0]) / 180)
    return np.arccos(a + b)*6371.0


def clusterClubs(numClust=5):
    datList = []
    with open('../data/KMeans/places.txt', 'r') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKMeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1,0.1,0.8,0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('../data/KMeans/Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    # 画出每一簇
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],\
                    ptsInCurrCluster[:,1].flatten().A[0],\
                    marker=markerStyle, s=90)
    # 标记出每一个蔟的centroid
    ax1.scatter(myCentroids[:,0].flatten().A[0],\
                myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()



if __name__ == '__main__':

    # 数据集 testSet.txt kMeans聚类
    # dat = loadDataSet('../data/KMeans/testSet.txt')
    # datArr = np.array(dat)
    # datMat = np.mat(dat)
    # myCentroids, clustAssing = kMeans(datMat, 4)
    # plotKMeans(datArr, 4, myCentroids, clustAssing)


    # 数据集 testSet2.txt
    # dat2 = loadDataSet('../data/KMeans/testSet2.txt')
    # datArr2 = np.array(dat2)
    # datMat2 = np.mat(dat2)
    # centList, myNewAssments = biKMeans(datMat2, 3)
    # plotKMeans(datArr2, 3, centList, myNewAssments)


    # 数据集 places.txt 对地理坐标进行聚类
    clusterClubs()