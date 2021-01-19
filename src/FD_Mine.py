from pandas import *
from collections import defaultdict
import sys


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 0)


def compute_no_trivial_closure(x, listofcol):
    global dictClosure
    xx = list(x)
    for item in xx:
        listofcol.remove(item)
    for y in listofcol:
        if y not in dictClosure[x]:
            if validfd(x, y):
                dictClosure[x].append(y)


def obtaint_FD_and_Key(x):
    global finallistofFDs
    global finallistofEQs
    global finallistofKEYs
    global dictClosure
    global listofcolumns
    for rh in dictClosure[x]:
        finallistofFDs.append([x, rh])
    if check_superkey(x):
        finallistofKEYs.append(x)


def obtain_EQSet(level, listofcols):
    global dictCplus
    global finallistofFDs
    global listofcolumns
    for x in level:
        xx = set(list(x))
        li = []
        for t in dictClosure.keys():
            if dictClosure[t]:
                li.append(t)
        for lh in li:
            if lh != x:
                z = xx.intersection(set(list(lh)))
                x_z = list(set(x) - set(z))
                x_z = ''.join(x_z)
                lh_z = list(set(lh) - set(z))
                lh_z = ''.join(lh_z)
                if x_z in dictClosure[lh] and lh_z in dictClosure[x]:
                    finallistofEQs.append([x, lh])


def validfd(x, y):
    if x == '' or y == '': return False
    ey = computeE(x)
    temp = x + y
    generate_closure(temp, x, y)
    stripped_product(temp, x, y)
    eyz = computeE(temp)
    if ey == eyz:
        return True
    else:
        return False


def computeE(x):
    global totaltuples
    global dictpartitions
    doublenorm = 0
    for i in dictpartitions[''.join(sorted(x))]:
        doublenorm = doublenorm + len(i)
    e = (doublenorm - len(dictpartitions[''.join(sorted(x))])) / float(totaltuples)
    return e


def check_superkey(x):
    global dictpartitions
    if (dictpartitions[x] == [[]]) or (dictpartitions[x] == []):
        return True
    else:
        return False


def prune_candidates(level):
    global finallistofEQs
    global finallistofKEYs
    for x in level:  # line 1
        if x in finallistofKEYs:
            level.remove(x)
        else:
            for y in level:
                if x != y and (([x, y] in finallistofEQs) or ([y, x] in finallistofEQs)):
                    level.remove(x)
                    break


def generate_closure(x, y, z):
    global dictClosure
    global dictpartitions
    temp = []
    if dictClosure[y]:
        for item in dictClosure[y]:
            temp.append(item)
    if dictClosure[z]:
        for item in dictClosure[z]:
            temp.append(item)
    temp = list(set(temp))
    dictClosure[x] = temp


def generate_candidates(level):
    global finallistofFDs
    global finallistofKEYs
    nextlevel = []
    for i in range(0, len(level)):  # pick an element
        for j in range(i + 1, len(level)):  # compare it to every element that comes after it.
            if ((not level[i] == level[j]) and level[i][0:-1] == level[j][0:-1]):  # i.e. line 2 and 3
                x = level[i] + level[j][-1]  # line 4
                flag = True
                for a in x:  # this entire for loop is for the 'for all' check in line 5
                    if not (x.replace(a, '') in level):
                        flag = False
                for k in x:
                    re = []
                    re = list(x)
                    re.remove(k)
                    re = ''.join(re)
                    if k in dictClosure[re]:
                        flag = False
                if flag == True:
                    stripped_product(x, level[i], level[
                        j])  # compute partition of x as pi_y * pi_z (where y is level[i] and z is level[j])
                    generate_closure(x, level[i], level[j])
                    if check_superkey(x):
                        finallistofKEYs.append(x)
                    else:
                        nextlevel.append(x)
    return nextlevel


def stripped_product(x, y, z):
    global dictpartitions
    global tableT
    tableS = [''] * len(tableT)
    partitionY = dictpartitions[''.join(sorted(y))]  # partitionY is a list of lists, each list is an equivalence class
    partitionZ = dictpartitions[''.join(sorted(z))]
    partitionofx = []  # line 1
    for i in range(len(partitionY)):
        for t in partitionY[i]:
            tableT[t] = i
        tableS[i] = ''
    for i in range(len(partitionZ)):
        for t in partitionZ[i]:
            if (not (tableT[t] == 'NULL')):
                tableS[tableT[t]] = sorted(list(set(tableS[tableT[t]]) | set([t])))
        for t in partitionZ[i]:
            if (not (tableT[t] == 'NULL')) and len(tableS[tableT[t]]) >= 2:
                partitionofx.append(tableS[tableT[t]])
            if not (tableT[t] == 'NULL'): tableS[tableT[t]] = ''
    for i in range(len(partitionY)):  # line 11
        for t in partitionY[i]:  # line 12
            tableT[t] = 'NULL'
    dictpartitions[''.join(sorted(x))] = partitionofx


def computeSingletonPartitions(listofcols):
    global data2D
    global dictpartitions
    for a in listofcols:
        dictpartitions[a] = []
        for element in list_duplicates(data2D[a].tolist()):
            # list_duplicates returns 2-tuples, where 1st is a value, and 2nd is a list of indices where that value occurs
            if len(element[1]) > 1:  # ignore singleton equivalence classes
                dictpartitions[a].append(element[1])


# ------------------------------------------------------- START ---------------------------------------------------


if __name__ == "__main__":
    infile = open('fundata.csv')
    data2D = read_csv(infile)
    totaltuples = len(data2D.index)
    listofcolumns = list(data2D.columns.values)  # returns ['A', 'B', 'C', 'D', .....]
    tableT = ['NULL'] * totaltuples  # this is for the table T used in the function stripped_product
    L0 = []
    dictClosure = defaultdict(list)
    for x in listofcolumns:
        dictClosure[x] = []
    dictpartitions = {}  # maps 'stringslikethis' to a list of lists, each of which contains indices
    computeSingletonPartitions(listofcolumns)
    finallistofFDs = []
    finallistofEQs = []
    finallistofKEYs = []
    L1 = listofcolumns[:]  # L1 is a copy of listofcolumns
    l = 1
    L = [L0, L1]
    while not (L[l] == []):
        for x in L[l]:
            compute_no_trivial_closure(x, listofcolumns[:])
            obtaint_FD_and_Key(x)
        obtain_EQSet(L[l], listofcolumns[:])
        prune_candidates(L[l])
        temp = generate_candidates(L[l])
        L.append(temp)
        l = l + 1
    print("List of all FDs: ", finallistofFDs)
    print("Total number of FDs found: ", len(finallistofFDs))
    print("List of all eq	s: ", finallistofEQs)
