import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


'''SVM
SVM 优化中一个特别好的地方就是，所有的运算都可以写成内积（看代码K[i,j]的意义之后理解）

核函数K(x,y)可以拆成 将输入空间（低维）映射到 特征空间（高维）,即由 x -> fai(x)
    但是，由于SMO算法中求解参数⍺时，用的时K(x,y)的值，而不需要拆开，因此不用考虑映射之后的向量是什么。
    常用的K(x,y)是：高斯径向核函数 Radial basis function kernel（RBF核）

对于最终用于分类的决策函数中，只需要支持向量来参与计算。

支持向量的数目存在一个最优值。
    如果支持向量太少，就可能得到一个很差的决策边界
    如果支持向量太多，也就相当于每次都利用整个数据集进行分类，这种分类方法成为泡影 K近邻 。
    
这里的SVM是一个二分类器，有很多是基于SVM构建多分类器的。
'''

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def plotDataSet(dataMat, labelMat):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(data_plus_np[:,0], data_plus_np[:,1], marker='s')
    plt.scatter(data_minus_np[:,0], data_minus_np[:,1], marker='o')
    plt.show()


def selectJrand(i, m):
    j = 1
    while j==i:
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

'''python numpy

方法：
    multiply(): 
        对于数组和矩阵操作相同。
        数组/矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
        multiply( scale, list) 对 数组/矩阵 中所有元素乘scale
    dot(): 
        数组：
        矩阵：
    *:
        数组：1、两个相同shape的数组，执行对应位置相乘，2、一个数组一个数值，使用广播broadcast，对数组中的每个元素乘以数值
        矩阵：1、两个矩阵相乘，满足(m,n)*(n,t) = (m,t)执行矩阵乘法，2、一个矩阵一个数值，使用广播broadcast，对矩阵中的每个元素乘以数值
    
    max():
        (a, axis=0,...)
        求序列的最值
    
    maximum():
        (X, Y, ...)
        X与Y逐位比较取大者
        
    nonzero(array):
        返回数组a中值不为零的元素的下标,返回值是一个长度为a.ndim(数组a的轴数)的元组
        np.nonzero(alphas.A>0)[0] 经常这样用来获得支持向量的下标
        
    mat.A
    mat.T
    mat[[1,2,3]]   获得一个matrix，有第一维下标为1,2,3的元素组成

python中内置 max():
    max(iterable, *[, default=obj, key=func]) -> value   max([1,2,3])
    max(arg1, arg2, *args, *[, key=func]) -> value       max(1,2,3)

'''


# SMO算法是一种启发式算法，如果所有变量的解都满足此最优化问题的KKT条件，那么这个最优化问题的解就得到了。

# 下面是对所有的变量进行两个for循环，遍历所有情况，没有使用启发式方法
def smoSimple(dataMat, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMat)
    labelMat = np.mat(classLabels).transpose() # (1,m).transpose = (m,1) 为了之后的向量话计算
    m,n = np.shape(dataMatrix)
    # 初始化参数
    b = 0
    alphas = np.mat(np.zeros((m,1)))
    iter = 0

    # 训练
    # 当迭代次数小于最大的迭代次数时（外循环）（只有在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，程序才会停止并退出while循环）
    while iter < maxIter:
        alphaPairsChanged = 0
        # 对数据集中的每个数据向量（内循环）
        for i in range(m):

            # 分离超平面 表达式
            # 预测值：（m,1).T * ( (m,n) * (1,n).T ) = 1 最后 +b 使用广播，得到训练数据i的根据当前分离超平面计算得到的预测值
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            '''
            dataMatrix[j,;] * dataMatrix[i,:].T  即数据向量j和数据向量i的内积，
            那么 dataMatrix * dataMatrix[i,:].T  得到一个向量，向量的第t个元素代表：数据向量t和数据向量i的内积
            '''

            # 误差：训练数据i预测值与真实值的误差
            Ei = fXi - float(labelMat[i])

            # 如果该数据向量可以被优化
            # 不管是正间隔还是负间隔都会被测试（其中正间隔是在当前分离超平面分类正确的）, 同时检查alpha的值，保证其不能等于0或者C
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):   # 这里是什么间隔？？？ 函数间隔不应该是y(w*x+b) 吗？
                # 如果参数是 0 / C 表示数据已经在 "边界" 上，不能够再减小或增大

                # 随机选择另外一个数据变量
                j = selectJrand(i, m)
                # 预测值：计算训练数据j根据当前 分离超平面 得到的预测值
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                # 误差：训练数据j预测值和真实值的误差
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()    # 深复制
                alphaJold = alphas[j].copy()

                # 两个变量的优化，根据变量的约束条件（不等式约束+等式约束），求得变量的取值范围。
                if labelMat[i] != labelMat[j]:  # 两个变量对应的两个样本的类标签不同
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:                           # 两个变量对应的两个样本的类标签相同
                    L = max(0, alphas[j] + alphas[i] -C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    print("L==H")
                    continue

                # 形式化为求解 凸二次规划 问题，子问题中的两个变量中只有一个是自由变量，另一个变量可以由自由变量表示
                # 沿着约束方向未经剪辑的解
                # 这里把 alphas[j] 当作自由变量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 求导关于alphas[j]导数为0，更新alphas[j]
                alphas[j] -= labelMat[j]*(Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)

                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue

                # 已知更新后的alphas[j], 更新 alphas[i]
                alphas[i] += labelMat[i]*labelMat[j]*(alphaJold - alphas[j])

                # 更新 b 值
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T

                if 0 < alphas[i] and  C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1+b2) / 2.0

                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))

        # 如果所有向量都没被优化，增加迭代次数，继续下一次循环。
        if alphaPairsChanged == 0: # 检査alpha值是否做了更新，如果有更新则将iter为0后继续运行程序 ？？？还是不明白为什么要置零
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)

    return b, alphas


def plotSupportVectors(dataMat, labelMat, alphas):
# def plotSupportVectors():
    # xcord0 = []
    # ycord0 = []
    # xcord1 = []
    # ycord1 = []
    # makers = []
    # colors = []
    #
    # with open("../data/SVM/testSet.txt", 'r') as fr:
    #     for line in fr.readlines():
    #         lineSplit = line.strip().split('\t')
    #         xPt = float(lineSplit[0])
    #         yPt = float(lineSplit[1])
    #         label = int(lineSplit[2])
    #         if label == -1:
    #             xcord0.append(xPt)
    #             ycord0.append(yPt)
    #         else:
    #             xcord1.append(xPt)
    #             ycord1.append(yPt)
    #
    # # 绘制所有样本
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # ax.scatter(xcord0, ycord0, marker='s', s=98)
    # ax.scatter(xcord0, ycord0, marker='s')
    # # ax.scatter(xcord1, ycord1, marker='o', s=98)
    # ax.scatter(xcord1, ycord1, marker='o')
    # plt.title("Support Vectors Circled")

    # 绘制所有样本
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(data_plus_np[:,0], data_plus_np[:,1], marker='s')
    plt.scatter(data_minus_np[:,0], data_minus_np[:,1], marker='o')



    # 绘制支持向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            # circle = Circle((x, y), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
            # ax.add_patch(circle)
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    # circle = Circle((4.6581910000000004, 3.507396), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
    # ax.add_patch(circle)
    # circle = Circle((3.4570959999999999, -0.082215999999999997), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
    # ax.add_patch(circle)
    # circle = Circle((6.0805730000000002, 0.41888599999999998), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
    # ax.add_patch(circle)
    # for i in sVect:
    #     circle = Circle(tuple(i), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
    #     ax.add_patch(circle)
    plt.show()


def plotResult(dataMat, labelMat, alphas, w, b):
    # 绘制所有样本
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(data_plus_np[:, 0], data_plus_np[:, 1], marker='s')
    plt.scatter(data_minus_np[:, 0], data_minus_np[:, 1], marker='o')

    # 绘制 分离直线
    x1 = max(dataMat)[0]  # max([1,2],[3,4]) 会根据元素的第一个位置的值比较，并返回整个元素
    x2 = min(dataMat)[0]
    # w shape为(n,1)
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2  # 分离平面为 w1*x1 + w2*x2 + b = 0
    plt.plot([x1, x2], [y1, y2])

    # 绘制支持向量
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()

def outputSupportVector(dataArr, labelArr, alphas):
    sVect = alphas[alphas>0] # 数组过滤，只对Numpy类型有用，首先得到一个满足条件的bool数组，再将这个数组应用到原始矩阵中，得到一个Numpy矩阵。
    sVect = []
    # print(np.shape(sVect))
    for i in range(100):
        if alphas[i] > 0.0:
            sVect.append(dataArr[i])
            print(dataArr[i], labelArr[i])


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2))) # 用于缓存误差，第一列是否有效标志，第二列实际的E值
        # 应用核函数
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i,:], kTup)

        '''
        对于线性核函数：
            K[i,j]：代表数据向量i和数据向量j的内积值
        
        对于高斯径向核函数：
            K[i,j]：代表数据向量i和数据向量j的高斯值
        '''


def calcEk(oS, k):
    """
    计算误差值并返回：根据当前的 分离超平面 计算训练数据k的预测值和真实值的 误差
    :param oS:
    :param k:
    :return:
    """
    # fXk = float(np.multiply(oS.alphas, oS.labelMat).T * (oS.X*oS.X[k,:].T) + oS.b)
    # 封装成核函数，注释掉的只是单一的线性核函数
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek= fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):
    """
    内循环中的启发式方法，选择具有最大步长的第二个变量j，选择标准是希望变量j有足够大的变化
    可以在 变量j更新 依赖于 |Ei-Ej|的，因此选择最大的 |Ei-Ej|
    :param i:
    :param oS:
    :param Ei:
    :return: 返回步长最大变量的下标j 以及 该变量对应的 误差 Ej
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # mat.A 代表将matrix转换为array, nonzero(array)[0] 取得array中元素非零的下标，
    if len(validEcacheList) > 1: # ? 为什么仅仅是在当前记录中选择最大的，可能有的数据没有被计算过啊？？？明明不是全局最大的
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei-Ek) # 变量j的更新，依赖于 |Ei-Ej|
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else: # 如果是第一次，标志为全为0，随机选取第二个变量j
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):
    """
    计算误差值，并存入缓存中
    :param oS:
    :param k:
    :return:
    """
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):  # 完整版内循环代码
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.labelMat[i]*Ei < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.labelMat[i] > 0)): # ??? 跟C有什么关系
        j,Ej = selectJ(i, oS, Ei)   # 启发式的选择第二个变量j
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        if oS.labelMat[i] != oS.labelMat[j]:  # 两个变量对应的两个样本的类标签不同
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:  # 两个变量对应的两个样本的类标签相同
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j,:] * oS.X[j,:].T   # 向量的内积
        # 封装成核函数，注释掉的只是单一的线性核函数
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0

        # 求导关于alphas[j]导数为0，更新alphas[j]
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)

        updateEk(oS, j) # 更新eCache

        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("j not moving enough")
            return 0

        # 已知更新后的alphas[j], 更新 alphas[i]
        oS.alphas[i] += oS.labelMat[i] * oS.labelMat[j] * (alphaJold - oS.alphas[j])

        updateEk(oS, i) # 更新eCache

        # 更新 b 值
        # b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
        #             oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        # b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
        #             oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        # 封装成核函数，注释掉的只是单一的线性核函数
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[j,j]

        if 0 < oS.alphas[i] and oS.C > oS.alphas[i]:
            oS.b = b1
        elif 0 < oS.alphas[j] and oS.C > oS.alphas[j]:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):   # 完整版外循环代码
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    # 当迭代次数超过指定最大值， 或者遍历整个集合都未对alpha对进行修改时，退出循环
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            # 遍历所有值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            # 遍历非边界值
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1

        # 每次迭代 根据情况更新entireSet值
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:    # ？？？ 我想的是，一旦这个条件成立，不是之后的所有迭代都没有意义吗？因为这个成立，说明上次迭代 所有的alpha都没有更新（发生改变），那下一次也不会发生改变啊？
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMat)
    w = np.zeros((n,1))
    # 大部分alpha值为0，最终起作用的只有支持向量
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i], dataMat[i,:].T)   # 根据学得的 alphas 求 分离超平面的 参数 w
    return w





def kernelTrans(X, A, kTup):
    """
    对输入特征向量进行核转换，K(x,y)
        这里只实现了两种转换：1.线性核函数 2.高斯径向核函数
    :param X:
    :param A:
    :param kTup:
    :return:
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    '''
    lin： linear，线性核函数，，
        <x,y> x和y的内积
        <X,y> 整个数据集X和y的内积，向量中每个元素x代表：向量x和y的内积
    rfb：径向核函数 K(x,y) ，高斯径向核函数
        K(x,y) x和y的高斯值
        K(X,y) 整个数据集X和y的高斯值，向量中每个元素x代表：向量x和y的高斯值
        
    下面是矩阵运算
    '''
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K / (-1*kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


def testRbf(k1=1.3):
    # 加载训练数据集，并训练
    dataArr, labelArr = loadDataSet('../data/SVM/testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()

    # 构建支持向量矩阵
    svInd = np.nonzero(alphas.A>0)[0]   # 非零元素下标
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])
    # 计算训练错误率
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], ('rbf', k1))    # 决策函数中，即分离超平面，只需要支持向量
        # （svNum, 1).T * (1, svNum) + b
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))

    print(sVs)
    plotSupportVectors(dataArr, labelArr, alphas)

    # 加载测试集，计算测试错误率
    dataArr, labelArr = loadDataSet('../data/SVM/testSetRBF2.txt')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


def img2vector(filename):
    returnVec = np.zeros((1, 1024))
    with open(filename, 'r') as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVec[0, 32*i+j] = int(lineStr[j])  # 一个一个赋值
    return returnVec

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2vector("%s/%s" % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    """
    用SVM构建 手写数字 的分类器， 由于实现的是二分类器，因此这里的训练集和测试集只有数字1和9
    :param kTu:
    :return:
    """

    # 加载训练数据集，并训练
    dataArr, labelArr = loadImages("../data/SVM/digits/trainingDigits")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()

    # 构建支持向量矩阵
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % np.shape(sVs)[0])

    # 计算训练错误率
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)  # 决策函数中，即分离超平面，只需要支持向量
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))


    # 加载测试集，计算测试错误率
    dataArr, labelArr = loadImages("../data/SVM/digits/testDigits")
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :], kTup)  # 决策函数中，即分离超平面，只需要支持向量
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))


if __name__ == '__main__':
    select = 3

    if select == 1:
        fileName = "../data/SVM/testSet.txt"
        dataArr, labelArr = loadDataSet(fileName)
        # print(dataArr)
        # print(labelArr)

        b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
        # b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)   # 就单在此数据集上，感觉这个效果没有simple的好
        # print(b)
        # print(alphas)

        outputSupportVector(dataArr, labelArr, alphas)
        w = calcWs(alphas, dataArr, labelArr)
        print(w)
        # 仅仅绘制支持向量
        plotSupportVectors(dataArr, labelArr, alphas)
        # 绘制分隔平面 和 支持向量
        plotResult(dataArr, labelArr, alphas, w, b)

    elif select == 2:
        testRbf()

    elif select == 3:
        testDigits(('rbf', 20))

    pass

