import sys
import numpy as np


def read_input(file):
    for line in file:
        yield line.rstrip()


input = read_input(sys.stdin)
input = [float(line) for line in input]
numInputs = len(input)
input = np.mat(input)
sqInput = np.power(input, 2)

# 均值和平方后的均值，用于计算全局的均值和方差
print("%d\t%f\t%f" % (numInputs, np.mean(input), np.mean(sqInput))) # 第一行是标准输出，也就是reduce的输入
print("report: still alive", file=sys.stderr)   # 第二行是 标准错误输出，即对主节点做出的响应报告，表明本节点工作正常。

"""
cat inputFile.txt | python mrMeanMapper.py
python mrMeanMapper.py < inputFile.txt
"""