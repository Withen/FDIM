'''
    bottom up of Fdep
'''

from pandas import *
from collections import defaultdict
import sys


def violated(a, b):
    global listofcolumns
    dis = []
    arg = []
    violated_fds = []
    for x in range(0, len(listofcolumns)):
        if a[x] == b[x]:
            arg.append(listofcolumns[x])
        else:
            dis.append(listofcolumns[x])
    str_arg = ''.join(arg)
    for a in dis:
        vfd = [str_arg,a]
        violated_fds.append(vfd)
    return violated_fds


def negative_cover(relation):
    NCOVER = []
    for i in range(0, len(relation)):
        for j in range(i+1, len(relation)):
            vfd = violated(relation[i], relation[j])
            for x in vfd:
                if x not in NCOVER:
                    NCOVER.append(x)
    return NCOVER


def initialise(R):
    deps = []
    for x in R:
        temp = ['']
        temp.append(x)
        deps.append(temp)
    return deps


def strcompare(s1, s2):
    if s1 == ['']:
        return True
    if len(s1)>len(s2):
        return False
    for x in s1:
        if x not in s2:
            return False
    return True


def entail(a, b):
    if strcompare(a[0],b[0]) and strcompare(b[1],a[1]):
        return True
    else:
        return False


def specialise(D, ND):
    global listofcolumns
    temp = listofcolumns.tolist()
    spec = []
    D_lhs = []
    for x in D[0]:
        if x != '':
            D_lhs.append(x)
    for x in ND[0]:
        if x in temp:
            temp.remove(str(x))
    for y in D[1]:
        if y in temp:
            temp.remove(str(y))
    for x in temp:
        t = D_lhs[:]  # 直接赋值的话是地址的复制，改变t的话也会改变D_lhs
        t.append(x)
        t.sort()
        lhs = ''.join(t)
        s = [lhs, D[1]]
        spec.append(s)
    return spec


def bottom_up(r):
    global listofcolumns
    DEPS = initialise(listofcolumns)
    NCOVER = negative_cover(r)
    for ND in NCOVER:
        TMP = DEPS[:]
        for D in DEPS:
            if entail(D, ND):
                TMP.remove(D)
                TMP += specialise(D, ND)
            DEPS = TMP[:]
    return DEPS


if __name__ == "__main__":
    infile = open('../data/iris.csv')
    # infile = open(sys.argv[1])
    data = read_csv(infile)
    ntuples = len(data.index)    # 元组数量
    listofcolumns = data.columns    # 属性列表,是index格式
    rules = bottom_up(data.values)
    for r in rules:
        print(r)
