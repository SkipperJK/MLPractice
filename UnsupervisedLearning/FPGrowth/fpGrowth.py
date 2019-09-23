import numpy as np

'''
搜索引擎自动补全查询词项

FP树：Frequent Pattern树， 频繁模式树

FP-growth算法能够更为高效的发现频繁项集（这里定义频繁项集是出现的次数，不是支持度，且代码中没有记录频繁项集的次数），但是不能用于发现关联规则
FP-growth算法只需要对数据库进行两次扫描，而Apriori算法对于每个潜在的频繁项集都会扫描数据集判定给定模式是否频繁（统计）

根据数据集，首先构建FP树，之后在FP树的基础上进行频繁项集挖掘。
    这里的代码是直接遍历所有的数据来构建FP树的。实际上可以重写createTree()函数，每次读入一个实例，并随着实例的不断输入而不断增长树。（？？？可以吗）
    FP-growth算法还有一个map-reduce版本的实现。


'''

'''python
for k in headerTable.keys():
    if headerTable[k] < minSup:
        del(headerTable[k])
会报错：dictionary changed size during iteration
因为在python3.x中 dict.keys()方法返回的是可迭代对象

更改为：for k in list(headerTable.keys()):
'''

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)


def createTree(dataSet, minSup=1):
    """
    构建FP树
    :param dataSet:
    :param minSup:
    :return:
    """
    headerTable = {}
    # 第一次遍历数据集，对整个数据集中的单类元素的个数进行统计
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 移除不满足最小支持度的元素项
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None   # 如果没有元素项满足要求，则退出

    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None) # 建立空集根节点
    # 第二次遍历数据集
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)] # 对数据集合中的元素根据单元素的频率进行排序
            updateTree(orderedItems, retTree, headerTable, count) # 使用排序后的频率项集对树进行扩充
    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children: # 先判断item[0]有没有在当前树的子树中
        inTree.children[items[0]].inc(count) # 如果有的话，更新count    （ 注意：不是直接根据第一遍的统计赋值的）
    else:
        # 如果没有在子树中，则创建子树，由于新创建树因此一定要更新链表
        inTree.children[items[0]] = treeNode(items[0], count, inTree) # 新建子树，并记录parent
        if headerTable[items[0]][1] == None: # 更新 headerTable 链表
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # 对该数据之后的元素，以前面的children[items[0]]为根，建立树，即对剩下的元素项迭代调用updateTree函数
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def ascendTree(leafNode, prefixPath):
    """
    迭代上溯整棵树
    :param leafNode:
    :param prefixPath:
    :return:
    """
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    condPats = {} # 条件模式基 conditional pattern base
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: # 记录长度超过1前缀路径
            condPats[frozenset(prefixPath[1:])] = treeNode.count # 该前缀路径的总数
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    """
    从一棵FP树中挖掘频繁项集
        递归函数，直到FP树只包含一个元素为止
    :param inTree:
    :param headerTable:
    :param minSup:
    :param preFix:
    :param freqItemList:
    :return:
    """
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1][0])] # ? p[1][0] # 从头指针表的底端开始
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)  # 从条件模式基来构建条件FP树
        if myHead != None:
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
            # 再使用 myHead，和newFreqSet递归调用mineTree()函数， 递归函数，直到FP树只包含一个元素为止




if __name__ == '__main__':

    # 使用手动创建的简单数据集构建FP树
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    myFPtree.disp()

    for key in myHeaderTab:
        print(key, myHeaderTab[key])

    xPrefixPath = findPrefixPath('x', myHeaderTab['x'][1])
    print(xPrefixPath)
    zPrefixPath = findPrefixPath('z', myHeaderTab['z'][1])
    print(zPrefixPath)
    rPrefixPath = findPrefixPath('r', myHeaderTab['r'][1])
    print(rPrefixPath)

    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print(freqItems)

    # 例子：从Twitter源中发现一些共现词


    # 例子：从新闻网站点击流中挖掘 数据集：kosarak.dat 100w条记录
    parseDat = [line.split() for line in open('../data/FPGrowth/kosarak.dat').readlines()]
    print(parseDat[0])
    initSet = createInitSet(parseDat)

    myFPtree, myHeaderTab = createTree(initSet, 100000)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
    print(len(myFreqList))
    print(myFreqList)

    # 应用：搜索引擎通过挖掘共现词自动补充