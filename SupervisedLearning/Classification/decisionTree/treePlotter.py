import matplotlib.pyplot as plt


# 定义节点格式常量
'''
boxstyle: 文本框的类型
fc: 边框粗细
'''
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制结点，annotate是关于一个数据点的文本标注
    :param nodeTxt: 要显示的文本
    :param centerPt: 文本的中心点，箭头所在的点，箭头终点
    :param parentPt: 指向文本的点，箭头起点
    :param nodeType:
    :return:
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )


'''python
dict.keys() 版本区别：
    python2.x: d.keys() 返回一个list
    python3.x: d.keys() 返回一个dict_keys对象，该对象不支持索引，如果要索引，可以转换成list(d.keys())


'''

def getNumLeafs(myTree):
    """
    计算叶子结点的总数
    :param myTree:
    :return:
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]   # 为树/子树的根结点，实际上这里的树的keys()永远只有一个
    print(myTree.keys())
    secondDict = myTree[firstStr]   # 树中的子结点
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])    # 如果当前树的子结点是内部结点，把这个子结点当作子树
        else:
            numLeafs += 1
    return numLeafs


def getNL(myTree):
    numLeafs = 0
    keys = myTree.keys()
    for key in keys:
        if type(myTree[key]).__name__ == 'dict':
            numLeafs += getNL(myTree[key])
        else:
            numLeafs += 1
    return numLeafs



def getTreeDepth(myTree):
    """
    计算树的深度
    :param myTree:
    :return:
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

'''
# 下面代码错误
def getTD(myTree):
    maxDepth = 0
    keys = myTree.keys()
    for key in keys:
        if type(myTree[key]).__name__ == 'dict':    # 如果使用这种方法，就会对每个子树都要+1，而不是取子树中最深的值
            thisDepth = 1 + getTD(myTree[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return thisDepth
'''


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    """
    绘制中间文本，即特征的取值
    :param cntrPt:
    :param parentPt:
    :param txtString:
    :return:
    """
    xMid = (parentPt[0] - cntrPt[0]) /2.0 + cntrPt[0]   # 求中间点的横坐标
    yMid = (parentPt[1] - cntrPt[1]) /2.0 + cntrPt[1]   # 求中间点的纵坐标
    createPlot.ax1.text(xMid, yMid, txtString)  # 绘制


def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制决策树， 代码太复杂了
    :param myTree:
    :param parentPt:
    :param nodeTxt:
    :return:
    """
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    """
    创建绘图
    :param inTree:
    :return:
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()   # 清空绘图区
    # createPlot.ax1 = plt.subplot(111, frameon=False)    # 注意：定义createPlot.ax1中的变量 ax1，这里定义的createPlot.ax1是全局变量
    # plotNode(u'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    # plotNode(u'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    # fig.show()
    axprops = dict(xticks=[], yticks=[])    # 定义横纵坐标
    # createPlot.ax1 为全局变量，绘制图像的句柄，subplot()定义了一个绘图，111表示figure中的图有1行1列，最后一个1代表第一个图，frameon即"frame on"表示是否绘制坐标轴矩形
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops) # python 中所有的变量默认都是全局有效的？？？？
    plotTree.totalW = float(getNumLeafs(inTree))    # 存储树的高度，全局变量
    plotTree.totalD = float(getTreeDepth(inTree))   # 存储树的深度，全局变量
    plotTree.xOff = -0.5 / plotTree.totalW  # 决策树的起始横坐标，全局变量
    plotTree.yOff = 1.0     # 决策树起始纵坐标，全局变量
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    # createPlot()
    myTree = retrieveTree(1)
    print(getNumLeafs(myTree))
    print(getNL(myTree))
    print(getTreeDepth(myTree))
    # print(getTD(myTree))
    createPlot(myTree)

    pass
