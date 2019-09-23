import numpy as np
"""
一共有三个例子：
    1 过滤网站的恶意留言
    2 过滤垃圾邮件
    3 从个人广告中获取区域倾向
"""


"""python
版本2 / 3 区别: Python 3 uses iterators for a lot of things where python 2 used lists，例如range()
    range(n):
        2.x: [1,2,...,n]
        3.x: iterator
        优点是如果使用大范围迭代器或映射，Python 3不需要分配内存
        
list 对象方法：
    list.count(element) 返回元素的个数
    list.remove(element) 删除指定的元素
"""


# 过滤网站的恶意留言
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 使用逻辑运算，取两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    词集模型 (set-of-words model)
    将每个词时候出现作为一个特征
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("The word: {} is not in my vocabulary!".format(word))
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    """
    词袋模型 (bag-of-words model)
    将每个词出现的次数作为一个特征
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
"""
 不对啊，这里的类条件概率只适用这个问题，虽然一个句子由一个向量表示，但是这个向量每一位都代表同一个特征啊
 这里的是把句子在词汇表中的每个词是否出现作为一个特征，这样理解的话，特征个数就是词汇表长度
 
 试想一下，训练数据集有两个特征，例如 vec = [ A, B ], 类别也有两个 labels = [ 0, 1 ]
    需要计算类条件概率：（个数应该为 numFeat * numLabel）
        p(A | 0) -> p(A = a1 | 0), p(A = a2 | 0),..., p(A = an | 0) 
        p(B | 0) -> p(B = b1 | 0), p(B = b2 | 0),..., p(B = bn | 0) 
        p(A | 1) -> p(A = a1 | 1), p(A = a2 | 1),..., p(A = an | 1) 
        p(B | 1) -> p(B = b1 | 1), p(B = b2 | 1),..., p(B = bn | 1) 
        
"""
# 命名，什么的数量  使用 numXXX
def trainNBO(trainMatrix, trainCategory):   # 训练算法
    """
    计算类条件概率，这里只处理二分类
    :param trainMatrix:
    :param trainCategory: 只有两类 0/1
    :return: 类条件概率，先验概率
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])  # 词汇集的大小，每个词对每个类都有一个类条件概率
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 计算类别为 1 侮辱性的文档的概率
    # 使用最大似然估计的话，可能发生类条件概率为0的情况，
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    # p0Denom = 0.0
    # p1Denom = 0.0

    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    # ??? 为什么要设置成2.0， 如果这样的话，类条件概率相加不为 1 啊，虽然影响不大
    # p0Denom = numWords
    # p1Denom = numWords
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] # 统计在该类别下，每个词出现的次数，数组相加
            p1Denom += sum(trainMatrix[i])  # 统计在该类别下所有词出现的总数, denomination 面值
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 这样计算得到的值太小，累乘时可能导致下溢出
    # p1Vect = p1Num / p1Denom    # 数组运算
    # p0Vect = p0Num / p0Denom    # 数组运算

    p1Vect = np.log(p1Num / p1Denom)    # 数组运算
    p0Vect = np.log(p0Num / p0Denom)    # 数组运算
    return p0Vect, p1Vect, pAbusive
# 在向量级别进行计算，而不是自己想的这种循环
'''
labels = set(instLabels)
classCount = [[0]*len(vocabList) for i in range(labels)]
for i, inst in enumerate(insts):
    for label in labels:
        if instLabels[i]  == label:
'''


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    求得后验概率，由于分母相同，没有必要计算，由于类条件概率取log之后，相乘运算变为 log相加运算
    :param vec2Classify:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


# 使用朴素贝叶斯过滤邮件
def textParse(bigString):
    """

    :param bigString:
    :return:
    """
    import re
    listOfTokens = re.split(r'\W*', bigString) # \W 任何非单词字符 === [^A-Za-z0-9_]  \w === [A-Za-z0-9_]
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 使用朴素贝叶斯过滤邮件
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open("../data/bayes/email/spam/%d.txt" % i, errors='ignore').read())    # 直接read
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open("../data/bayes/email/ham/%d.txt" % i, errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)) # 在python2.x下range() return list, 而在python3.x下 range() return iterator
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet))) # 根据random，随机划分训练集和测试集，留存交叉验证
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat= []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNBO(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('The error rate is: {}'.format(float(errorCount) / len(testSet))) # 这里只用了一次迭代，如果更精确的估计分类器的错误率，可以多次迭代之后求平均错误率


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    # sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    sortedFreq = sorted(freqDict.items(), key=lambda d: d[1], reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.extend(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = np.random.uniform(0, len(trainingSet))
        testSet.append((trainingSet[randIndex]))
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNBO(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('The error rate is: {}'.format(float(errorCount) / len(testSet)))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -0.6:
            topSF.append(vocabList[i], p0V[i])
        if p1V[i] > -0.6:
            topNY.append(vocabList[i], p1V[i])
    sortedSP = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**")
    for item in sortedSP:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item(0))

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNBO(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print("{} classified as: {}".format(str(testEntry), classifyNB(thisDoc, p0V, p1V, pAb)))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print("{} classified as: {}".format(str(testEntry), classifyNB(thisDoc, p0V, p1V, pAb)))

if __name__ == '__main__':
    # listOPosts, listClasses = loadDataSet()
    # myVocabList = createVocabList(listOPosts)
    # print(myVocabList)
    # print(setOfWords2Vec(myVocabList, listOPosts[0]))
    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # p0V, p1V, pAb = trainNBO(trainMat, listClasses)
    # print(p0V, p1V, pAb)
    # print(sum(p0V), sum(p1V))

    # testingNB()
    # print(textParse("I ammmm aaaaa student"))
    spamTest()