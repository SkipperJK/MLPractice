import numpy as np


"""
Apriori原理是说如果一个元素项是不频繁的，那么哪些包含该元素的超集也是不频繁的。Apriori算法从单元素项集开始，通过组合满足最小支持度要求的项集来形成更大的集合。
缺点：每次增加频繁项集的大小，Apriori算法都会重新扫描整个数据集。
"""

'''python
set()和frozenset()
    set无序排序且不重复，是可变的，有add(), remove()等方法。既然是可变的，所以它不存在哈希值。
    frozenset是冻结的集合，是不可变的，存在哈希值，好处是它可以作为字典的key，也可以作为其他元素的集合。缺点是一旦创建便不能更改，没有add(), remove()方法。
    
    集合的方法 子集 s0.issubset(s1),  超集 s1.issuperset(s1)

'''

def loadDataSet():
    return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]


def createC1(dataSet):
    """
    生成所有的单个物品的项集列表
    :param dataSet:
    :return:
    """
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    """
    计算D中的所有频繁子项，并记录频繁子项的支持度
    :param D:
    :param Ck:
    :param minSupport:
    :return:
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                # D中项的子集项数量也+1
                if not can in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    # 计算所有项集的支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):
    """
    根据上一个的项集合，来生成下一项的所有项集
    :param Lk:
    :param k:
    :return:
    """
    retList = []
    lenLK = len(Lk)
    for i in range(lenLK):
        for j in range(i+1, lenLK):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            # *** 为了不得到并集之后重复的结果，只选择合并两个集合的前面k-2个元素相等的集合
            if L1 == L2:
                retList.append(Lk[i] | Lk[j]) # | 为集合合并操作符
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1] # 所有项集的列表
    k = 2
    while(len(L[k-2]) > 0): # 只要上一级有元素集合，就根据上一级的元素集合生成下一级的项集合 （频繁项集）
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK) # 把supK 中的键值更新到supportData中 （update操作如果有相同的key则被替换）
        L.append(Lk)
        k += 1
    return L, supportData


def generateRules(L, supportData, minConf=0.7):
    """
    首先从一个频繁项集开始，接着创建一个规则列表，其中规则右部只包含一个元素，然后对这些规则进行测试。接下来合并所有剩余规则来创建一个新的规则列表，其中规则右部包含两个元素。
    :param L:
    :param supportData:
    :param minConf:
    :return:
    """
    bigRuleList = []
    # 由于无法从 单元素项集 中构建关联规则，因此要从包含两个或多个元素的项集开始构建规则
    for i in range(1, len(L)): # 对每个频繁项集list（物品个数相同的频繁项集在同一列表中） i从1开始，即从L[1]包含两个元素的项集开始
        for freqSet in L[i]:  # 对项集list中的每个频繁项集
            H1 = [frozenset([item]) for item in freqSet] # 对 每个 频繁项集创建只包含 单个元素集合 的list
            print("freqSet", freqSet)
            print("元素list", H1)
            if (i > 1): # 如果频繁项集中的元素超过2个，那么会考虑合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


def calcConf(freqSet, H, supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        print("计算规则置信度：", freqSet, conseq, conf)
        if conf >= minConf:
            print(freqSet-conseq, '--->', conseq, 'conf', conf)
            br1.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0]) # 这个不应该恒为1吗？  不是1，因为该函数是迭代函数，最原始的项集元素的个数是1，
    if (len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H, m+1)  # 根据当前项集元素的个数为m，生成元素个数为m+1的项集
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if (len(Hmp1) > 1): # 如果新生成的项集个数不为0
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)


if __name__ == '__main__':

    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    D = list(map(set,dataSet)) # 将数据转换成集合格式，从而可以使用 issubset(), issuperset()方法
    print(dataSet)
    L1, suppData0 = scanD(D, C1, 0.5) # 在所有数据中计算所有 当个物品项集C1 的 频繁项集
    print(L1)
    #
    # # 支持度 = 0。5
    L, suppData = apriori(dataSet)
    print(L)
    #
    # # 支持度 = 0。7
    L, suppData = apriori(dataSet, 0.7)
    print(L)
    #
    #
    L, suppData = apriori(dataSet, minSupport=0.5)
    print(L)
    print(suppData)
    rules = generateRules(L, suppData, minConf=0.7)
    print(rules)
    rules = generateRules(L, suppData, minConf=0.5)
    print(rules)


    # 真实例子：搜索引擎中的查询词


    # 真实例子：发现毒蘑菇的相似特征
    # 用来挖掘毒蘑菇中存在的公共特征，样本数据中，每个样本都有23个特征，每个特征有不同个取值个数，这里把样本的每个特征的所有取值的编号都不同，从而寻找频繁项集
    mushDatSet = [line.split() for line in open('../data/Apriori/mushroom.dat').readlines()]
    L, suppData = apriori(mushDatSet, minSupport=0.3)
    # 第一个特征代表蘑菇是否有毒，如果有毒，值为2
    for item in L[1]:
        if item.intersection('2'):
            print(item)
    for item in L[3]:
        if item.intersection('2'):
            print(item)
