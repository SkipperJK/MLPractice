from numpy import *
import operator
import os
'''numpy
numpy库中存在两种不同的数据类型（矩阵matrix和矩阵array），都可以处理行列表示的数字元素。
两个数据类型分别为：
    numpy.ndarray： array() 多维数组
    numpy.matrix：  mat()    矩阵

numpy 中常用的方法：
    zeros(shape)
    ones(shape)
    eye(shape)
    random.random(shape)
    
    sum(ndarray[,axis=]) 如果不指定axis参数，则对多维数组所有元素进行加操作
    tile(A, reps) 根据给定的reps重复A，构建一个array，这的reps可以是int 或者 tuple
    min(axis=) 
    max(axis=)
    
array**2  对数组对象中的每个元素取平方，（操作符是对数组中的逐个元素处理的）


array对象
    属性
    array.ndim 数组的维度个数
    array.shape 数组的维度，返回元组
    array.size 数组中包含的元素个数
    array.dtype 数组中元素的数据类型
    
    方法
    array.sum(axis=0)
    array.argsort() 返回的是数组值从小到大的索引值, 如果要降序，-array
        argsort(a, axis=-1, kind='quicksort', order=None)
            根据指定的axis排序，  axis=0 #按列排序，axis=1 #按行排序
            
    
理解参数axis，指的就是多维数组中shape的第几维，
    例如:
        array.sum(axis=0) 就是不同行的元素相加，
        array.sum(axis=1) 就是不同列的元素相加，
        array.min(axis=0) 在行的维度，挑选出所有行中每一列最小的
        array.min(axis=1) 在列的维度，挑选出所有列中每一行最小的
    通过不同的axis，numpy会沿着不同的方向进行操作：如果不设置，那么对所有的元素操作；
    如果axis=0，则沿着纵轴进行操作；axis=1，则沿着横轴进行操作。
    设axis=i，则numpy沿着第i个**下标变化的方向**进行操作。


    
        
    
'''

'''python
dict.get(key, default=None)

在python2中：
    dict.items(): Return a copy of the dictionary’s list of (key, value) pairs.
    dict.iteritems(): Return an iterator over the dictionary’s (key, value) pairs.
在python3中：
    dict.items(): Return an iterator 
    dict.iteritems() 已经弃用
    
    
    
Python内置的排序函数
    sort 是应用在 list 上的方法
    sorted 可以对所有可迭代的对象进行排序操作
        sorted(iterable[, cmp[, key[, reverse]]])
        iterable -- 可迭代对象
        cmp -- 比较的函数
        key -- 主要用来进行比较的元素，可以多个
        reverse -- True降序，False升序（默认）
        例子：
        
        list:
            L=[('b',2),('a',1),('c',3),('d',4)]
            # sorted(L, cmp=lambda x,y:cmp(x[1], y[1]))
            sorted(L, key=lambda x: x[1])
        
        dict:
            D = {'b':2, 'a':1, 'c':3, 'd':4}
            # sorted(D.items(),  cmp=lambda x,y:cmp(x[1], y[1]))
            sorted(D.items(), key=lambda x: x[1])
            
            students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
            sorted(students, key=lambda s: s[2])
            
            NOTE: dict 也是可迭代对象，不过默认迭代的是key：
                [i for i in dict] = ['b', 'a', 'c', 'd']
                [i for i in dict.keys()] = ['b', 'a', 'c', 'd']
                [i for i in dict.values()] = [2, 1, 3, 4]
                [i for i in dict.items()] = [('b', 2), ('a', 1), ('c', 3), ('d', 4)]
                
        return 排序之后的list
                
                
判断对象是否是可迭代对象
    from collections import Iterable
    isinstance(obj, Iterable)           
                
'''

'''operator
    operator.itemgetter() 函数用于获取对象的哪些维的数据
'''

def createDateSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    基于欧式距离的kNN分类器
    :param inX:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 使用矩阵运算，整个矩阵的所有元素都减去待分类元素，之后再计算距离
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1  # 代码简洁
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 按降序
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    根据数据集的大小，先构建大小维度相同的以0填充的多维数组，在读取文件中的值进行填充
    :param filename:
    :return:
    """
    with open(filename, 'r') as fr:
        arrayOLines = fr.readlines()    # 返回list
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()     # 默认去除字符串 头尾 的空白符， 此处的空格包含'\n', '\r',  '\t',  ' '
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]     # 将读取出来的值，赋值给多维数组
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    对样本所有特征数据进行归一化处理，这里归一到 0-1 范围
    由于在计算欧式距离的时候，如果所有特征不归一化，对于数值差距较大的特征对距离计算结果影响较大
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    将训练数据集进行训练集和测试集划分，并计算分类器的准确率
    :return:
    """
    hoRatio = 0.50
    datingDataMat, datingLabels = file2matrix("../data/kNN/datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)    # 划分测试集，取前numTestVecs个
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 1)
        print("Predicate: {}, Real: {}".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("The total error rate is {}".format(errorCount/float(numTestVecs)))
    print(errorCount)


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    precentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix("../data/kNN/datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, precentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this persion: {}".format(resultList[classifierResult - 1]))


def img2vector(filename):
    """
    这里是把图片存储为文本文件，图片都是统一格式的32*32（这里如果是真的图像处理的话，应该是先把图片背景转换成黑色，只有手写的数字为像素快为白色，在使用像素块的值作为向量值（要归一化？））
    :param filename:
    :return:
    """
    returnVec = zeros((1, 1024))
    with open(filename, 'r') as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVec[0, 32*i+j] = int(lineStr[j])  # 一个一个赋值
    return returnVec


def handwritingClassTest():
    hwLabels = []
    trainingFilePath = "/Users/skipper/Project/Learn/MLProject/data/kNN/digits/trainingDigits"
    trainingFileList = [os.path.join(item[0], file) for item in os.walk(trainingFilePath) for file in item[2]]
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))  # 初始化训练集矩阵
    for i in range(m):
        filePathStr = trainingFileList[i]
        fileNameStr = filePathStr.split('/')[-1]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        hwLabels.append((classNumStr))
        trainingMat[i, :] = img2vector(filePathStr)
    testFilePath = "/Users/skipper/Project/Learn/MLProject/data/kNN/digits/trainingDigits"
    testFileList = [os.path.join(item[0], file) for item in os.walk(testFilePath) for file in item[2]]
    mTest = len(testFileList)
    errorCount = 1.0
    for i in range(mTest):
        filePathStr = testFileList[i]
        fileNameStr = filePathStr.split('/')[-1]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        vectorUnderTest = img2vector(filePathStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("Predicate: {}, Real: {}".format(classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("The total error rate is {}".format(errorCount / float(mTest)))
    print(errorCount)


def main():
    filename = '../data/kNN/datingTestSet2.txt'
    datingDataMat,  datingLabels = file2matrix(filename)
    print(datingDataMat, datingLabels)
    datingClassTest()
    return datingDataMat,  datingLabels
'''matplotlib
pyplot文件：
    方法：
        figure() 返回以个Figure类的实例

Figure类：
    方法：
        add_subplot()
        
scatter(x,y,s=20,c='b',maker='o'[,...]) 
    
'''

if __name__ == '__main__':
    # datingDataMat,  datingLabels = main()
    # import matplotlib
    # import matplotlib.pyplot as plt
    # fig = plt.figure()  # 定义figure
    # ax = fig.add_subplot(111)   # 定义subplot
    # # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
    #            15.0*array(datingLabels), 15.0*array(datingLabels))  # 使用特征2，3绘图
    # ay = fig.add_subplot(222)
    # ay.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
    #            15.0*array(datingLabels), 15.0*array(datingLabels))  # 使用特征1，2绘图
    # plt.show()
    # classifyPerson()

    # testVector = img2vector("../data/kNN/digits/trainingDigits/0_0.txt")
    # print(testVector[0, 0:31])  # 注意，这里返回的是ndarray多维数组类型，即使只有一行，也要使用多维下标访问

    handwritingClassTest()



'''
Summary:
    这里只是简单的实现了k近邻分类，使用遍历的方法计算与训练集中每个样本的距离。如果使用kd树来存储数据可以减少计算量。
    kNN算法首先有要足够的标记样本数据得到的效果才比较好。
    KNN算法是基于实例的学习，使用算法是必须有接近实际数据的训练样本数据。
    无法给出任何数据的基础结构信息，因此无法知晓实例样本和典型实例样本具有什么特征。
'''