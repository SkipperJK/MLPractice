from math import log
import pickle
from SupervisedLearning.Classification.decisionTree.treePlotter import *

"""DecisionTree
这是使用的是ID3算法，选择信息增益最大的特征作为最优特征，没有进行剪枝操作，会overfitting
    ID3算法无法直接处理 数值型 数据，而且会倾向与选择取值较多的特征
"""

"""python
pickle 模块可以序列化对象
    序列化对象可以保存到磁盘上，需要时可以从磁盘上读取
    python中任何对象都可以执行序列化操作
    
    主要函数：
        dump(obj, file[, protocol]) #序列化到文件
        load(file) 
        dumps(obj) # 序列化到内存（字符串格式保存），然后对象可以以任何方式处理如通过网络传输，return 一个字符串对象
        loads(obj) # 将字符串反序列化为对象

json
    主要函数
        dump(obj, fp,...)    Serialize obj as a JSON formatted stream to fp
        load(fp [,encoding[,xxx]]) Deserialize fp to a Python object
        dumps(obj, ...) Serialize obj to a JSON formatted str
        loads(s [,encoding [,xxx]]) Deserialize s(str or unicode instance) to a Python object
        
"""


"""algorithm
对于递归函数：
    一定要清楚递归的结束条件是什么，清楚return是什么（可能没有return）
    
看人家写出来的代码，对于其他的数据也可以复用，因此在写代码时，考虑到函数的复用，要决定函数参数的选择
"""

'''theory
计算某个特征X对数据集D带来的信息增益（本质上是D和X的互信息）
g(D, A) = H(D) - H(D|A)
先计算数据集D的信息熵，然后计算在基于特征A条件的数据集D的信息熵，即条件熵 H(D|A)
'''

'''code
    calcShannonEnt() 计算当前数据集的信息熵，就是D的信息熵， H(D) = - \sum_{i}^{K} p_{i} log(p_{i})  其中K是输出的类别个数
    splitDataSet()  根据指定的特征所在的axis对数据集划分，value是指指定特征的取值，return split之后的数据集，为指定特征为value的样本，且样本中不在有该特征
    chooseBestFeatureToSplit() 对当前所有特征计算特征增益，对样本计算H(D)，然后对每个特征计算 H(D|A)，找出信息增益最大的特征，返回该特征的所在列下标
'''

def calcShannonEnt(dataSet):
    """
    计算训练数据集D的信息熵，输出类别个数即D的类别个数
    :param dataSet: 训练数据集，每个样本为一个向量，包含特征和类别标签，向量最后一维是类别标签
    :return:
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 最后一个样本的类别标签
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    # 注意：如果使用np.array()下面中的整型值都会变成字符
    # dataSet = np.array([[1, 1, 'yes'],
    #                  [1, 1, 'yes'],
    #                  [1, 0, 'no'],
    #                  [0, 1, 'no'],
    #                  [0, 1, 'no']])
    dataSet = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']   # 特征的label，第一维是"no surfacing"，第二维是"flippers"
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    根据指定的feature和其值对数据集进行划分，返回指定feature为指定value的样本，注意：划分之后的样本的features不再包含当前feature
    :param dataSet:
    :param axis: 指定feature的轴
    :param value: 指定feature的值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:  # 只记录指定的feature和其值，并删除当前特征所在的列
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    在当前数据集上，根据信息增益选择信息增益最大的特征并进行分割
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 得到某一特征的所有取值情况
        newEntropy = 0.0
        # 对于指定的特征计算 条件熵 H(D|A)
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)  # 对于特征A的所有取值计算H(D|A = a_i)再求期望，得到H(D|A)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    在划分的过程中，由于使用ID3算法，可能出现剩余的训练集没有特征了，只剩下类别标记，因此对于这个叶子结点的类别采用多数表决的方法
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    # sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    sortedClassCount = sorted(classCount.items(), key=lambda d: d[1], reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    根据训练数据集构造决策树（应该首先把伪算法写出来，要考虑到所有情况）
    其中两个 if 判断，就是递归的连个停止条件：1.所有实例属于一类，2.没有特征
    :param dataSet:
    :param labels: 是特征的label，不是输出类别
    :return:
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):     # 如果训练数据集都属于一类
        return classList[0]
    if len(dataSet[0]) == 1:    # 如果没有特征，多数表决决定类别，其中dataSet[i]代表第i个样本
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)    # 找出当前训练集信心增益最大的特征
    bestFeatLable = labels[bestFeat]     # 记录最优特征的label
    myTree = {bestFeatLable: {}}    # 使用dict表示树
    del labels[bestFeat]    # 从labels中删除该特征
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)    # 记录最优特征的所有取值情况
    for value in uniqueVals:    # 按照选择特征的取值个数c，递归创建 c 个子树
        subLabels = labels[:]
        myTree[bestFeatLable][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)    # 递归构建树
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    对输入的数据根据在训练集上构建好的决策树进行分类
    :param inputTree:
    :param featLabels:
    :param testVec:
    :return:
    """
    firstStr = list(inputTree.keys())[0]    # 这里的keys()实际上只有一个，就是当前树的根结点
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # 获取当前结点的特征在向量中的下标
    for key in secondDict.keys():
        if testVec[featIndex] == key:   # 根据树中根结点代表的特征取值，判断指向树的那个分支，如果分支是叶结点即输出类别，如果分支是非叶结点则递归判断
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:    # 在python2中，可以使用'w'，why？？？
        pickle.dump(inputTree, fw)  # 将数据对象 dump 到文件中，fw为文件句柄


def grabTree(filename):
    with open(filename, 'rb') as fr:
        return pickle.load(fr)  # 从文件中 load 数据对象，fr为文件句柄


def lense():
    with open("../data/decisionTree/lenses.txt", 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lensesTree = createTree(lenses, lensesLabels)
        createPlot(lensesTree)


if __name__ == '__main__':

    # myDat, labels = createDataSet()
    # print(labels)
    # shannonEnt = calcShannonEnt(myDat)
    # print(shannonEnt)
    # retDataSet = splitDataSet(myDat, 0, 0)
    # print(retDataSet)
    # bestFeature = chooseBestFeatureToSplit(myDat)
    # print(bestFeature)
    # myTree = createTree(myDat, labels)  # 注意，在这里 labels变量被修改
    #
    # labels = ['no surfacing', 'flippers']
    # print(classify(myTree, labels, [1,0]))
    # print(classify(myTree, labels, [1,1]))
    #
    # storeTree(myTree, "../data/decisionTree/classifierStorage.txt")
    # myTree1 = grabTree("../data/decisionTree/classifierStorage.txt")
    # print(myTree1)

    lense()
    pass

