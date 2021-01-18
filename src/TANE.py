# -*- coding: utf-8 -*

import sys
import time
from itertools import combinations
from functools import reduce


def read_db(path):
    hashes = {}    # partitions
    with open(path, 'r') as fin:
        line0 = fin.readline().strip()
        global attrset
        attrset = line0.split(',')
        print("attribte set:", attrset)
        for t, line in enumerate(fin):  # t为行序数, line为行数据
            line = line.strip()    # 去除首尾空格
            if line == '':
                break
            for i, s in enumerate(line.split(',')):  # i为属性序数, s为属性值
                # setdefault为字典存入不存在的关键字; set()函数创建一个无序不重复元素集; t是set()中的行号数据, 使左手边可多属性
                hashes.setdefault(i, {}).setdefault(s, set([])).add(t)  # [(i, s)] = len(hashes)
        # print(hashes)
        # 计算剥离分区
        strippedPartition = [PPattern.fix_desc(hashes[k].values()) for k in sorted(hashes.keys())]
        print("strippedPartition", strippedPartition)
        return strippedPartition


def toStr(atts):
    return ''.join([chr(65 + i) for i in atts])


class PPattern(object):
    '''
    Represents the Stripped Partition
    '''

    _top = None
    _bottom = None

    n_elements = 0

    @classmethod
    def intersection(cls, desc1, desc2):
        '''
        Procedure STRIPPED_PRODUCT defined in [1]
        '''
        new_desc = []
        T = {}
        S = {}
        for i, k in enumerate(desc1):
            for t in k:
                T[t] = i
            S[i] = set([])
        for i, k in enumerate(desc2):
            for t in k:
                if T.get(t, None) is not None:
                    S[T[t]].add(t)
            for t in k:
                if T.get(t, None) is not None:
                    if len(S[T[t]]) > 1:
                        new_desc.append(S[T[t]])
                    S[T[t]] = set([])
        return new_desc

    @classmethod
    def fix_desc(cls, desc):
        n_elements = sum([len(i) for i in desc])
        if cls.n_elements < n_elements:
            cls.n_elements = n_elements
        for i in range(len(desc) - 1, -1, -1):
            if len(desc[i]) == 1:
                del desc[i]
        return cls.sort_description(desc)

    @classmethod
    def sort_description(cls, desc):
        desc.sort(key=lambda x: (len(x), sorted(x)), reverse=True)
        return desc

    @classmethod
    def leq(cls, desc1, desc2):
        if desc1 == cls._bottom:
            return True
        for i in desc1:
            check = False
            for j in desc2:
                if len(i) > len(j):
                    break
                if i.issubset(j):
                    check = True
                    break
            if not check:
                return False
        return True


class PartitionsManager(object):

    # current_level 当前层级
    # cache[current_level] 属性集
    # cache[current_level][X] X的剥离分区

    def __init__(self, T):
        '''
        Initializes the cache
        '''
        self.T = T  # T为计算好的剥离分区
        # self.cache = {0: None, 1: {(i,): j for i, j in zip(attr, T)}}  # 属性集：剥离分区
        self.cache = {0: None, 1: {(i,): j for i, j in enumerate(T)}}  # 属性集：剥离分区
        self.current_level = 1

    def new_level(self):
        '''
        Creates a cache for the new level
        '''
        self.current_level += 1
        self.cache[self.current_level] = {}

    def purge_old_level(self):
        '''
        Memory wipe of unused cache
        '''
        del self.cache[self.current_level - 2]

    def register_partition(self, X, X0, X1):
        '''
        Registers partition of attributes in X, using partitions
        already calculated of attributes in X0 and X1
        '''
        self.cache[len(X)][X] = PPattern.intersection(self.cache[len(X0)][X0], self.cache[len(X1)][X1])
        # print(X)
        # print(self.cache[len(X)])
        # print(self.cache[len(X)][X])

    def check_fd(self, X, y):
        '''
        Main difference with [1], we do not check using procedure "e"
        to check and FD, but we use partition subsumption
        Seems more efficient
        '''
        if not bool(X):
            return False
        left = self.cache[len(X)][X]
        return PPattern.leq(left, self.T[y])  # 判断left是否T[y]的精化（精化则存在推出关系）

    def is_superkey(self, X):
        # 超键：在X上任何两个元组都不是完全相等的 → X的剥离分区为空集
        return not bool(self.cache[len(X)][X])


class rdict(dict):
    '''
    Recursive dictionary implementing Cplus
    '''

    def __init__(self, *args, **kwargs):
        super(rdict, self).__init__(*args, **kwargs)
        self.itemlist = super(rdict, self).keys()

    def __getitem__(self, key):  # 可根据key获取value，即rdict[key]
        if key not in self:
            self[key] = self.recursive_search(key)
        return super(rdict, self).__getitem__(key)

    def recursive_search(self, key):
        result = reduce(set.intersection, [self[tuple(key[:i] + key[i + 1:])] for i in range(len(key))])  # 计算最大子集的C+集交集
        return result


def calculate_e(X, XA, R, checker):
    '''
    计算错误度量e，X和XA为剥离分区，R为全部属性，checker为
    '''
    e = 0
    T = {}
    if not bool(X):  # 剥离分区为空，X是超键
        return -1
    X = checker.cache[len(X)][X]
    XA = checker.cache[len(XA)][XA]

    for c in XA:
        T[next(iter(c))] = len(c)  # 为c生成迭代器，next进行取值
    for c in X:
        m = 1
        for t in c:
            m = max(m, T.get(t, 0))  # 无此值T[t]则输出0(不设置)
        e += len(c) - m
    return float(e) / len(R)


def prefix_blocks(L):
    blocks = {}
    for atts in L:
        blocks.setdefault(atts[:-1], []).append(atts)  # setfault 设置当前key的value
    return blocks.values()


class TANE(object):
    def __init__(self, T):
        self.T = T
        self.rules = []

        self.pmgr = PartitionsManager(T)
        self.R = range(len(T))  # R = range(0, x)

        self.Cplus = rdict()  # Cplus的类型为rdict
        self.Cplus[tuple([])] = set(self.R)  # L0的C+初始化为R(set)

    def compute_dependencies(self, L):
        for X in L:
            for y in self.Cplus[X].intersection(X):    # 计算在C+集和X集交集中的属性
                a = X.index(y)    # 获取y在X中的下标
                LHS = X[:a] + X[a + 1:]    # 左手边，最大子集X\{A}
                if self.pmgr.check_fd(LHS, y):    # 验证X\{A} → A
                    self.rulesappend(LHS, y)
                    # self.rules.append((LHS, y))
                    self.Cplus[X].remove(y)    # 移除A
                    map(self.Cplus[X].remove, filter(lambda i: i not in X, self.Cplus[X]))    # 移除C+(X)\X

    def rulesappend(self, LHS, RHS):
        X = ()
        for lhs in LHS:
            X = X + tuple([attrset[lhs]])
        A = attrset[RHS]
        self.rules.append((X, A))

    def prune(self, L):
        '''
        Procedure PRUNE described in [1]
        '''
        clean_idx = set([])
        for X in L:
            if not bool(self.Cplus[X]):    # C+集为空
                clean_idx.add(X)
            if self.pmgr.is_superkey(X):    # Is Superkey, since it's a stripped partition, then it's an empty set
                for y in filter(lambda x: x not in X, self.Cplus[X]):    # C+(X)\X
                    if y in reduce(set.intersection,
                                   [self.Cplus[tuple(sorted(X[:b] + X[b + 1:] + (y,)))] for b in range(len(X))]):    # 判断X为主键
                        # self.rules.append((X, y))
                        self.rulesappend(X, y)
                clean_idx.add(X)
        for X in clean_idx:
            L.remove(X)

    def prefix_blocks(self, L):
        '''
        Procedure PREFIX_BLOCKS described in [1]
        '''
        blocks = {}
        for atts in L:
            blocks.setdefault(atts[:-1], []).append(atts)
        return blocks.values()

    def generate_next_level(self, L):
        '''
        Procedure GENERATE_NEXT_LEVEL described in [1]
        '''
        self.pmgr.new_level()
        next_L = set([])
        for k in prefix_blocks(L):
            for i, j in combinations(k, 2):
                if i[-1] < j[-1]:
                    X = i + (j[-1],)
                else:
                    X = j + (i[-1],)
                # if all(X[:a] + X[a + 1:] in L for a, x in enumerate(X)):
                if all(X[:a] + X[a + 1:] in L for a in range(len(X))):
                    next_L.add(X)
                    # WE ADD THIS LINE, SEEMS A BETTER ALTERNATIVE TO CALCULATE THE PARTITION HERE WHEN
                    # WE HAVE REFERENCES TO BOTH PARTITIONS USED TO CALCULATE IT
                    self.pmgr.register_partition(X, i, j)
        return next_L

    def memory_wipe(self):
        '''
        FREE SOME MEMORY!!!
        '''
        self.pmgr.purge_old_level()

    def run(self):
        '''
        Procedure TANE in [1]
        '''
        L1 = set([tuple([i]) for i in self.R])
        L = [None, L1]
        l = 1
        while bool(L[l]):
            self.compute_dependencies(L[l])
            self.prune(L[l])
            L.append(self.generate_next_level(L[l]))
            l = l + 1
            # MEMORY WIPE
            L[l - 1] = None
            self.memory_wipe()


if __name__ == "__main__":
    T = read_db(sys.argv[1])
    tane = TANE(T)
    t0 = time.time()
    tane.run()
    print("\t=> Execution Time: {} seconds".format(time.time() - t0))
    print("\t=> {} Rules Found".format(len(tane.rules)))
    for item in tane.rules:
        print(item)

'''
    PCDBERCa
    P → CDBERCa
    C → PDBERCa
    E → BCa
    R → Ca
    BR → E
    BCa → E
    DBR → PC
    DER → PC
'''
