import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
import numpy as np
from SupervisedLearning.Regression.regTrees import *


"""matplotlib
scatter() 方法构建的是离散型散点图
plot() 方法构建的是连续曲线

"""


def reDraw(tolS, tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2
        myTree = createTree(reDraw.rawDat, modelLeaf, modelErr, (tolS, tolN))
        yHat = createForeCast(myTree, reDraw.testDat, modelTreeEval)
    else:
        myTree = createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = createForeCast(myTree, reDraw.testDat)
    reDraw.a.scatter(reDraw.rawDat[:,0].A1, reDraw.rawDat[:,1].A1, s=5)
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    # reDraw.canvas.show()
    reDraw.canvas.draw()
    # show() 使用 draw() 替换


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')
    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')
    return tolN, tolS


def drawNewTree():
    tolN, tolS = getInputs()
    reDraw(tolS, tolN)



root = Tk()

Label(root, text='Plot Place Holder').grid(row=0, columnspan=3)
reDraw.f = Figure(figsize=(5,4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
# reDraw.canvas.show()
reDraw.canvas.draw()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text='tolN').grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')
Label(root, text='tolS').grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')
Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text='Model Tree', variable = chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)
reDraw.rawDat = np.mat(loadDataSet('../data/TreeRegression/sine.txt'))
reDraw.testDat = np.arange(min(reDraw.rawDat[:,0]), max(reDraw.rawDat[:,0]), 0.01)
reDraw(1.0, 10)
root.mainloop()