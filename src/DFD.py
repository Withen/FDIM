# -*- coding: utf-8 -*

import sys
import time
import copy
from collections import defaultdict
from math import ceil

class BitSet:#修改自https://github.com/rbotarleanu/Python-Bitset
    """
    A lightweight bitset class which can hold a fixed-size idx of bits in a
    memory-efficient manner.
    Methods
    -------
    set(position=-1, value=1)
        Set either all bits (when position is -1) or a certain bit
        (position>=0) to value.

    reset(position=-1)
        Set all bits (when position is -1) or the position bit to 0.
    flip(position=-1)
        Flips the value of the bit at *position*. If *position* is -1, all bits
        are flipped.

    Example
    -------
        bs = BitSet(7)  # 0000000
        bs.set()  # 1111111
        bs.flip()  # 0000000
        bs.set(0, 1)  # 1000000
        bs.set(5, 1)  # 1000010
        bs.flip(3)  # 1001010
        bit_value = bs[3]  # 1
    """

    __INT_SIZE = 32
    __ALL_SET = (1 << 32) - 1
    __ALL_CLEAR = 0

    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.__bits = [self.__ALL_CLEAR
                       for _ in range(self.__number_of_integers_needed(n_bits))]

    def __number_of_integers_needed(self, n_bits):
        """
        Computes the number of integers required to store n_bits.
        Parameters
        ----------
        n_bits: int
            the number of bits to be stored.
        """

        return int(ceil(n_bits / self.__INT_SIZE))

    def __index_bit_in_bitset(self, position):
        """
        Computes the index in the bitset array that holds the *position* bit.
        Parameters
        ----------
        position: int
            the position of the bit in the bitset.
        Returns
        -------
        tuple
            the index of the corresponding idx in the bitset array and the
            index of the bit in that idx.
        """
        return divmod(position, self.__INT_SIZE)

    def __clear(self, idx, bit):
        """
        Clears the value of the *bit* bit of the *idx* integer.
        Parameters
        ----------
        idx: int
            index of the integer in the array holding the bits.
        bit: int
            index of the bit of that integer
        """
        if bit < 0 or bit > self.n_bits:
            raise ValueError("Bit position should not exceed BitSet capacity.")
        self.__bits[idx] &= ~(1 << bit)

    def __set(self, idx, bit):
        """
        Sets the value of the *bit* bit of the *idx* integer.
        Parameters
        ----------
        idx: int
            index of the integer in the array holding the bits.
        bit: int
            index of the bit of that integer
        """
        if bit < 0 or bit > self.n_bits:
            raise ValueError("Bit position should not exceed BitSet capacity.")

        self.__bits[idx] |= (1 << bit)

    def __flip(self, idx, bit):
        """
        Flips the value of the *bit* bit of the *idx* integer. As such, 0
        becomes 1 and vice-versa.
        Parameters
        ----------
        idx: int
            index of the integer in the array holding the bits.
        bit: int
            index of the bit of that integer
        """
        if bit < 0 or bit > self.n_bits:
            raise ValueError("Bit position should not exceed BitSet capacity.")

        self.__bits[idx] ^= (1 << bit)

    def __get(self, idx, bit):
        """
        Gets the value of the *bit* bit of the *idx* integer.
        Parameters
        ----------
        idx: int
            index of the integer in the array holding the bits.
        bit: int
            index of the bit of that integer
        """
        if bit < 0 or bit > self.n_bits:
            raise ValueError("Bit position should not exceed BitSet capacity.")

        return int(self.__bits[idx] & (1 << bit) > 0)

    def cardi(self):
        count=0
        for i in list(range(self.n_bits)):
            idx, bit = self.__index_bit_in_bitset(i)
            if self.__get(idx,bit)==1:
                count+=1
        return count
    def size(self):
        return self.n_bits

    def set(self, position=-1, value=1):
        """
        Sets the bit at *position* to *value*.
        If *position* is -1, all bits are set to *value*.
        Parameters
        ----------
        position: int
            the position at which to perform the set operation.
        value: int
            the value to use when setting the bit.
        """

        if position == -1:
            mask = self.__ALL_SET if value == 1 else self.__ALL_CLEAR
            for i in range(len(self.__bits)):
                self.__bits[i] = mask
        else:
            idx, bit = self.__index_bit_in_bitset(position)
            if bit < 0 or bit > self.n_bits:
                raise ValueError("Bit position should not exceed BitSet "
                                 "capacity.")

            if value == 1:
                self.__set(idx, bit)
            else:
                self.__clear(idx, bit)

    def reset(self, position=-1):
        """
        Resets the bit at *position* to 0.
        If *position* is -1, all bits are set to 0.
        Parameters
        ----------
        position: int
            the position at which to perform the set operation.
        """

        self.set(position, value=0)

    def __flip_all(self):
        """
        Flips the values of all bits in the bitset.
        """
        for i in range(len(self.__bits)):
            self.__bits[i] ^= self.__ALL_SET

    def flip(self, position=-1):
        """
        Flips the bit at *position* such that 0 becomes 1 and vice-versa.
        If *position* is -1, all bits are flippsed.
        Parameters
        ----------
        position: int
            the position at which to perform the set operation.
        """

        if position == -1:
            self.__flip_all()
            return

        idx, bit = self.__index_bit_in_bitset(position)
        if bit < 0 or bit > self.n_bits:
            raise ValueError("Bit position should not exceed BitSet capacity.")

        self.__flip(idx, bit)

    def __getitem__(self, position):
        idx, bit = self.__index_bit_in_bitset(position)
        if bit < 0 or bit > self.n_bits:
            raise ValueError("Bit position should not exceed BitSet capacity.")

        return self.__get(idx, bit)

    def __int_to_bitstring(self, idx):
        """
        Converts the integer at position idx to a string of 0's and 1's.
        Parameters
        ----------
        idx: int
            an index in the __bits array.

        Returns
        -------
        str
            a string of 1's and 0's
        """
        bitstring = ""
        for i in range(0, self.__INT_SIZE):
            if idx == len(self.__bits) - 1 and \
                    i >= self.n_bits % self.__INT_SIZE:
                break
            bitstring += str(self.__get(idx, i))

        return bitstring[:self.n_bits]

    def __str__(self):
        s = ""

        for i in range(len(self.__bits)):
            s += self.__int_to_bitstring(i)

        return s

    def _union(self,other):
        for i in range(other.size()):
            if other[i]==1:
                self.flip(i)

class attr_combination(BitSet):
    def __init__(self, capacity):
        BitSet.__init__(self, capacity)

    def set_copy(self,index,value):
        cop=copy.deepcopy(self)
        cop.set(index,value)
        return cop
    def is_atomic(self):
        if self.cardi()!=1:
            return 0
        else:
            return self.cardi()
    '''
    def add_attr(self,column_index):
        copy=self
        copy.set(column_index,1)
        return copy
    '''
    def or_copy(self,colle):
        bits=set(colle.get_bits())-set(self.get_bits())
        cop=copy.deepcopy(self)
        [cop.set(i,1) for i in bits]
        return cop
    def __eq__(self, other):
        return self.n_bits==other.n_bits and self.get_bits()==other.get_bits()
    def __hash__(self):
        return hash(str(self))
    def complement(self,all,dele):
        '''

        Parameters
        ----------
        all--当前level 0的所有属性 list<int>
        dele--补集中需要删去的 list<int>

        Returns--colle
        -------

        '''
        cop = copy.deepcopy(self)
        cop.flip()
        for i in range(cop.n_bits):
            if not i in all:
                cop.set(i,0)
        for i in dele:
            cop.set(i,0)
        return cop
    '''
        def complement(self):
        copy = self
        copy.flip()
        return copy
    '''
    def is_subset(self,other):
        #u=self.or_copy(other)
        #i=u.cardi()
        #j=other.cardi()
        return self.or_copy(other).cardi()==other.cardi()
    def is_superset(self,other):
        return self.or_copy(other).cardi() == self.cardi()
    def union(self,other):
        cop=copy.deepcopy(self)
        cop._union(other)
        return cop
    def get_bits(self):
        list=[]
        for i in range(self.n_bits):
            if self[i]==1:
                list.append(i)
        return list

def read_db(path):
    '''
    (借用学姐的)
    Parameters
    ----------
    path--file path

    Returns--list of partitions for candidates of one attribute
    -------

    '''
    hashes = {}    # partitions
    with open(path, 'r') as fin:
        line0 = fin.readline().strip()
        global attrset #从文件读入的属性集合
        attrset = line0.split(',')
        print("attribte set:", attrset)
        global num_of_rows  # 总元组数目
        num_of_rows=0
        for t, line in enumerate(fin):  # t为行序数, line为行数据
            num_of_rows+=1
            line = line.strip()    # 去除首尾空格
            if line == '':
                break
            for i, s in enumerate(line.split(',')):  # i为属性序数, s为属性值
                # setdefault为字典存入不存在的关键字; set()函数创建一个无序不重复元素集; t是set()中的行号数据, 使左手边可多属性
                hashes.setdefault(i, {}).setdefault(s, set([])).add(t)  # [(i, s)] = len(hashes)
        # print(hashes)
        # 计算剥离分区
        #list_values = [i for i in d1.values()]
        strippedPartition = [PPattern.fix_desc(hashes[k].values()) for k in sorted(hashes.keys())]
        # print("strippedPartition", strippedPartition)
        #global mm_partitions
        #mm_partitions = MM_partitions(strippedPartition.size)
        #for i in range(attrset.size):
            #mm_partitions.add_partition(strippedPartition.get(i))
        return strippedPartition # list<set<list<int>>>

class PPattern(object):
    '''
    Represents the Stripped Partition
    (借用学姐的)
    '''

    _top = None
    _bottom = None

    n_elements = 0

    @classmethod
    def fix_desc(cls, desc):
        desc=list(desc)
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

def int_to_indices(i): #index->indices
    colle=attr_combination(len(attrset))
    colle.set(i,1)
    return colle

def list_to_indices(l): #list<index>->indices
    colle=attr_combination(len(attrset))
    for i in l:
        colle.set(i, 1)
    return colle

def colle_to_attr(colle):#indices->string(candidates)
    str_attr=''
    for i in colle.get_bits():
        str_attr+=attrset[i]
    return str_attr

class DFD(object):
    def __init__(self, path):
        '''
        Parameters
        ----------
        path  file path
        '''
        self.T = read_db(path)#计算好的单个属性的剥离分区 dict<int,list<list<int>>>
        self.KEY = []#list<int>  数据集上的主键的index
        self.ATTR = list(range(len(attrset)))#[1,2,3...] 单个属性的index
        self.RHS=self.ATTR#list<int>  进行findLHS()的每个右手边属性的index
        self.FD = defaultdict(list)#最终输出的fd集合 dict<int,list<colle>>  右手边：左手边列表
    def count(self):
        sum=0
        for it in self.FD.values():
            sum+=len(it)
        return sum

    def is_unique(self):
        '''
        剔除主键
        check each column for uniqueness
		if a column is unique it's a key for all other columns
		therefore uniquePartition -> schema - uniquePartition
        '''
        for i,tp in enumerate(self.T):#i是属性序号 int，tp是属性对应的剥离分区 list<list<int>>
            if len(tp)== 0:
                self.KEY.append(i)
        self.RHS=list(set(self.RHS).difference(set(self.KEY)))#去掉key之后的属性索引
        for key in self.RHS:
            for l in self.KEY:
                self.FD[key].append(int_to_indices(l))#主键->其他每个属性  add unique columns to minimal uniques

    def run(self):
        '''
        The main loop of DFD
        '''
        global mm_partitions #存储剥离分区
        mm_partitions = MM_partitions(len(attrset))#dict<cardi,dict<colle,partition>>
        mm_partitions.setdefault(1,{})
        #先把从文件读取到的单个属性的partition存入mm_partitions
        for i, s in enumerate(self.T):#i是属性index，s是剥离分区 self.T--初始单个属性的剥离分区
            p=Partition(i,len(attrset),num_of_rows)
            colle=int_to_indices(i)
            p.clear()
            for l in s:
                p.add(frozenset(l))
            mm_partitions.get(1)[colle]=p

        #begin outer loop
        self.is_unique()#先剔除主键，对剩下的属性分别建立搜索空间
        for rhs in self.RHS:#对每个右手边建立搜索空间，查找左手边
            lhs=findLHS(rhs,self.RHS)#index list<int>  do this for all RHS
            l_lhs=lhs.run()#list<colle> 返回一个左手边列表
            for l in l_lhs:
                self.FD[rhs].append(l)


class Cate(dict):
    '''
    six categories for node and some method to categorizing node
    '''
    DEPENDENCY=1
    MINIMAL_DEPENDENCY=2
    CANDIDATE_MINIMAL_DEPENDENCY=3
    NON_DEPENDENCY=4
    MAXIMAL_NON_DEPENDENCY=5
    CANDIDATE_MAXIMAL_NON_DEPENDENCY = 6
    @classmethod
    def is_candidate(self,cate):
        return any([cate==self.CANDIDATE_MAXIMAL_NON_DEPENDENCY ,cate == self.CANDIDATE_MINIMAL_DEPENDENCY])

    @classmethod
    def is_dependency(self, cate):
        return any([cate==self.DEPENDENCY , cate==self.MINIMAL_DEPENDENCY , cate==self.CANDIDATE_MINIMAL_DEPENDENCY])

    @classmethod
    def is_non_dependency(self, cate):
        return any([cate==self.NON_DEPENDENCY , cate==self.MAXIMAL_NON_DEPENDENCY , cate==self.CANDIDATE_MAXIMAL_NON_DEPENDENCY])

    def unchecked_maximal_subsets(self,node):#返回list<colle>
        subsets=[]
        for i in node.get_bits():
            sub=node.set_copy(i,0)
            if not self.keys().__contains__(sub):
                subsets.append(sub) #colle of unchecked_maximal_subset
        return subsets
    '''
    def unchecked_maximal_subset(self,node):
        lhs=node
        for i in node.indices.get_bits():
            sub=lhs.set(i,0)
            if not self.keys().__contains__(sub):
                return sub#colle of unchecked_maximal_subset
        return None
    '''
    def unchecked_minimal_supersets(self,node,all,rhs):#返回list<colle>
        supersets=[]
        lhs=node.complement(all,[rhs])
        for i in lhs.get_bits():
            sub=node.set_copy(i,1)
            if not self.keys().__contains__(sub):
                supersets.append(sub)#colle of unchecked_minimal_superset
        return supersets
    '''
    def unchecked_minimal_superset(self,node,rhs):
        lhs=node.indices.complement()
        lhs=lhs.set(rhs,0)
        for i in lhs.get_bits():
            sub=lhs.set(i,1)
            if not self.keys().__contains__(sub):
                return sub#colle of unchecked_minimal_superset
        return None
    '''
    def update_d(self,lhs):
        '''
        判断该node(isDependency)的类别是否需要更新
        Parameters
        ----------
        lhs--node(isDependency) to update

        Returns--类别
        -------

        '''
        if lhs.cardi()>1:#如果存在子集
            flag=0
            for i in lhs.get_bits():
                cate=self.get(lhs.set_copy(i,0))#检查其所有少一个属性的子集
                if cate==None:#存在子集未被访问过
                    flag=1
                    break
                elif self.is_dependency(cate):#子集是fd
                    return self.DEPENDENCY
            if flag==1:
                return self.CANDIDATE_MINIMAL_DEPENDENCY
        return self.MINIMAL_DEPENDENCY
    def update_n(self,lhs,all,rhs):
        '''
        判断该node(is non-Dependency)的类别是否需要更新
        Parameters
        ----------
        lhs--node(is non-Dependency) to update
        rhs--当前右手边index

        Returns--类别
        -------

        '''
        flag=0
        colle=lhs.complement(all,[rhs])
        for i in colle.get_bits():
            cate=self.get(lhs.set_copy(i,1))#检查所有多一个属性的超集
            if cate==None:#存在超集未被访问
                flag=1
                break
            elif self.is_non_dependency(cate):#超集是non-fd
                  return self.NON_DEPENDENCY
        if flag==1:
            return self.CANDIDATE_MAXIMAL_NON_DEPENDENCY
        return self.MAXIMAL_NON_DEPENDENCY

class findLHS(object):
    def __init__(self,rhs,R):
        '''
        Parameters
        ----------
        rhs--int 当前的右手边的index
        R--list<int> 用于查找左手边的属性集合（包括当前右手边在内）
        '''
        self.RHS=rhs#int 当前的右手边的index
        self.eles = copy.deepcopy(R)
        self.eles.remove(rhs) #list<int> 用于构建搜索空间的属性索引
        self.D=[]#list<colle>  minimal dependencies 最后返回的值（返回一个左手边列表）
        self.ND=[]#list<colle>  maximal non-dependencies
        self.seeds=[]#list<colle>
        #colle即node
        self.dependencies = Dependencies(self.eles)# 3.5 dict<colle,set<colle>>
        self.non_dependencies = Non_dependencies(self.eles)# 3.5 dict<colle,set<colle>>
        self.trace=[]#stack<colle> 记录遍历路径，用于回溯
        self.observations=Cate()#dict(节点colle，类别)

    def run(self):
        '''

        Dfd determines all minimal Fds for the current Rhs attribute by classifying all possible Lhss.

        Returns--list<colle> 返回一个左手边列表
        -------

        '''
        for i in self.eles:#generate seeds     seeds ← R \ {A}
            seed=int_to_indices(i)
            self.seeds.append(seed)
        while len(self.seeds) != 0:#Algorithm 2: findLHSs() line 2
            while len(self.seeds) != 0:#Algorithm 2: findLHSs() line 4
                node = self.seeds.pop()  # node ← pickSeed();
                while node is not None:
                    self.trace.append(node)
                    if self.observations.keys().__contains__(node):  # if visited(node)
                        cate = self.observations[node]
                        if Cate.is_candidate(cate):  # if isCandidate(node)
                            if Cate.is_dependency(cate):#判断是否需要更新
                                update = self.observations.update_d(node)#类别
                                self.observations[node] = update
                                if not Cate.is_candidate(update):
                                    self.trace.pop()
                                if update == Cate.MINIMAL_DEPENDENCY:#if we couldn't find any dependency that is a subset of the current valid LHS it is minimal
                                    self.D.append(node)
                                    #self.trace.pop()
                            else:
                                update = self.observations.update_n(node,self.eles,self.RHS)#类别
                                self.observations[node] = update
                                if not Cate.is_candidate(update):
                                    self.trace.pop()
                                if update == Cate.MAXIMAL_NON_DEPENDENCY:#// if we couldn't find any non-dependency that is superset of the current non-valid LHS it is maximal
                                    self.ND.append(node)
                                    #self.trace.pop()
                        else:  self.trace.pop()
                    else:#未访问过
                        cate = self.check(node,self.RHS)#判断类别（推断或者计算分区）
                        if cate == Cate.MINIMAL_DEPENDENCY:  # if we couldn't find any dependency that is a subset of the current valid LHS it is minimal
                            self.D.append(node)
                        if cate == Cate.MAXIMAL_NON_DEPENDENCY:  # // if we couldn't find any non-dependency that is superset of the current non-valid LHS it is maximal
                            self.ND.append(node)

                    node = self.pick_next_node(node)#Algorithm 2: line18
            self.seeds = self.generate_seed()#Algorithm 2: line20 generate possibly remaining candidates
        return self.D#list<colle> 返回一个左手边列表

    def check(self,node,rhs):#未访问过的node
        '''

        Parameters
        ----------
        node-节点node
        rhs-右手边index

        Returns-cate 判断得到的节点类别
        -------

        '''

        # inferCategory(node)尝试根据已分类节点来判断该结点的类别
        if self.non_dependencies.inferred(node):#可从已有non-fd推断出类别
            cate=self.observations.update_n(node,self.eles,self.RHS)
            self.observations[node] = cate
            self.non_dependencies.add(node)
        elif self.dependencies.inferred(node):#可从已有fd推断出类别
            cate = self.observations.update_d(node)
            self.observations[node] = cate
            self.dependencies.add(node)
        #需要计算剥离分区
        c_rhs_partition=mm_partitions.get_p(int_to_indices(rhs))

        if node.is_atomic():
            c_lhs_partition=mm_partitions.get_p(node)
            c_joined_partition=Composed_partition(c_lhs_partition,c_rhs_partition)
            mm_partitions.add_partition(c_joined_partition)
        else:
            '''
            if we went upwards in the lattice we can build the currentLHS
            partition directly from the previous partition
            '''
            '''
            if node.addi!=-1:
                attr=node.addi#index
                p_lhs_partition=mm_partitions.get_p(node.base)
                if p_lhs_partition==None:
                    l_partitons=mm_partitions.get_matching(node.base)#list of partitons
                    p_lhs_partition=Composed_partition.build_p(l_partitons)
                addi_partiton=mm_partitions.get_p(int_to_indices(attr))
                c_lhs_partition=mm_partitions.get_p(p_lhs_partition.indices.set_copy(attr,1))
                if c_lhs_partition==None:
                    c_lhs_partition=Composed_partition(p_lhs_partition,addi_partiton)
                    mm_partitions.add_partition(c_lhs_partition)
                c_joined_partition=mm_partitions.get_p(c_lhs_partition.indices.set_copy(rhs,1))
                if c_joined_partition==None:
                    c_joined_partition=Composed_partition(c_lhs_partition,c_rhs_partition)
                    mm_partitions.add_partition(c_joined_partition)
            else:
            '''
            c_lhs_partition=mm_partitions.get_p(node)
            if c_lhs_partition==None:
                l_partitons = mm_partitions.get_matching(node)  # list of partitons
                c_lhs_partition = Composed_partition.build_p(l_partitons)#partition
                mm_partitions.add_partition(c_lhs_partition)
            c_joined_partition=mm_partitions.get_p(c_lhs_partition.indices.set_copy(rhs,1))
            if c_joined_partition == None:
                c_joined_partition = Composed_partition(c_lhs_partition, c_rhs_partition)
                mm_partitions.add_partition(c_joined_partition)
        #c_lhs_partition是左手边partition,c_joined_partition是左手边+右手边partition
        if Partition.representsFD(c_lhs_partition, c_joined_partition):#判断是否是fd
            cate=self.observations.update_d(node)
            self.observations[node]=cate
            self.dependencies.add(node)
            return cate
        cate = self.observations.update_n(node,self.eles,rhs)
        self.observations[node] = cate
        self.non_dependencies.add(node)
        return cate

    def generate_seed(self):#Algorithm 4: generateNextSeeds()
        '''

        generate possibly remaining candidates

        Returns--新的seed集合  list<node>
        -------

        '''
        seeds=[]#list<colle>
        newseeds=[]#list<colle>
        for max in self.ND:
            compl=max.complement(self.eles,[self.RHS]) #Algorithm 4 line 4
            if len(seeds)==0:
                for i in compl.get_bits():
                    seeds.append(int_to_indices(i))
            else:
                for s in seeds:
                    for i in compl.get_bits():
                        newseeds.append(s.set_copy(i,1))
                #minimize newseeds
                minimized=self.minimize_seeds(newseeds)#line 13
                seeds.clear()
                seeds=minimized
                newseeds.clear()
        #return only elements that aren't already covered by the minimal dependencies
        seeds=list(set(seeds)-set(self.D))
        return seeds#list<node>

    def minimize_seeds(self,seeds):#list<colle>
        '''

        for seed in seeds[:]:
            if self.observations.keys.contains(seed):
                seeds.remove(seed)

        '''
        max_cardi=0
        devided_seeds={}#dict<int,list<colle>> 按照cardi给seeds分组
        for seed in seeds:#colle
            key=seed.cardi()
            max_cardi=max(max_cardi,key)
            devided_seeds.setdefault(key,[])
            devided_seeds[key].append(seed)

        for i in range(max_cardi):
            lower=devided_seeds.get(i)
            if lower !=None:
                j=max_cardi
                while(j>i):
                    upper=devided_seeds.get(j)
                    if upper != None:
                       # lowerit=iter(lower)
                       #upperit=iter(upper)
                        for l_seed in lower:
                            for u_seed in upper:
                                if l_seed.is_subset(u_seed):
                                    upper.remove(u_seed)
                    j-=1
        minimized=set()
        for seedslist in devided_seeds.values():
            for s in seedslist:
                minimized.add(s)
        return minimized#list<colle>

    def pick_next_node(self,current_node):#Algorithm 3: pickNextNode()
        '''
        Dfd picks the next node based on its stack-trace and the currently considered column com-bination.
        Parameters
        ----------
        current_node--the currently considered column com-bination

        Returns--返回一个node
        -------

        '''
        cate=self.observations[current_node]
        if cate==self.observations.CANDIDATE_MINIMAL_DEPENDENCY:#向下retrieve
            s=self.observations.unchecked_maximal_subsets(current_node)#list<colle>
            p=self.non_dependencies.pruned_supersets(s)#list<colle>
            for it in p:
                self.observations[it]=self.observations.NON_DEPENDENCY
            s=list(set(s).difference(set(p)))#Algorithm 3 line 4
            if len(s)==0:
                self.observations[current_node]=self.observations.MINIMAL_DEPENDENCY
                self.D.append(current_node)
            else:
                node=s.pop()
                #self.trace.append(node)
                return node
        elif cate==self.observations.CANDIDATE_MAXIMAL_NON_DEPENDENCY:#向上retrieve
            s = self.observations.unchecked_minimal_supersets(current_node,self.eles,self.RHS)#list<colle>
            pn=self.non_dependencies.pruned_supersets(s)#list<colle>
            pd=self.dependencies.pruned_subsets(s)#list<colle>
            for it in pn:
                self.observations[it]=self.observations.NON_DEPENDENCY
            for it in pd:
                self.observations[it]=self.observations.DEPENDENCY
            s = list(set(s).difference(set(pn)))#Algorithm 3  line 14
            s = list(set(s).difference(set(pd)))
            if len(s)==0:
                self.observations[current_node]=self.observations.MAXIMAL_NON_DEPENDENCY
                self.ND.append(current_node)
            else:
                #n=s.pop()
                node=s.pop()
                #self.trace.append(node)
                return node
        if len(self.trace)!=0:#回溯 直接返回trace中latest one
            next=self.trace.pop()
            return next
        return None

'''
class Node(object):#？？是不是直接用colle代替掉
    def __init__(self, colle,addi=-1):# attr_combinarion int
        if addi!=-1:
            self.indices =colle.set(addi)
            self.base=colle
        else:
            self.indices=colle
            self.base = None
        self.addi=addi#int
    def is_atomic(self):
        return self.indices.cardi()==1
    def __eq__(self, other):
        return (self.indices,self.base,self.addi)==(other.indices,other.base,other.addi)
    def __hash__(self):
        return hash((self.indices,self.base,self.addi))
'''

class Partition(set):  # set<set<int>>
    probe_table=None
    def __init__(self, base, addi, f=-1):#int columnIndex, int numberOfColumns, int numberOfRows
        super().__init__()
        if f==-1:#partition partition
            self.indices = base.indices.or_copy(addi.indices)
            self.e = -1
            self.num_rows = base.num_rows
            self.dis=-1
            if Partition.probe_table==None:
                Partition.probe_table=[]
                for i in range(self.num_rows+1):
                    Partition.probe_table.append(-1)
        else:#int int int
            self.indices=attr_combination(addi)
            self.indices.set(base)
            self.num_rows=f
            self.e = -1
            self.dis = -1
            if Partition.probe_table==None or len(Partition.probe_table)!=self.num_rows:
                Partition.probe_table = []
                for i in range(self.num_rows+1):
                    Partition.probe_table.append(-1)

    def reset_probetable(self):
        for i in range(Partition.probe_table.size()):
            Partition.probe_table[i]=-1
    def get_e(self):
        if self.e==-1:
            c=0
            for eq in self:
                c+=len(eq)
            er=(c-len(self))/self.num_rows
            self.e=er
        return self.e
    @classmethod
    def representsFD(self,base,joined):
        return base.get_e()==joined.get_e()
    '''

    def equals(self,other):#？？？没用
        index=0
        num=0
        for eq in self:
            for it in iter(eq):
                Partition.probe_table[it] = index
                num+=1
            index+=1
        for eq in other:
            index=-2
            for it in iter(eq):
                c_index=Partition.probe_table[it]
                if index==-2 or index==c_index:
                    index=c_index
                else:
                    self.reset_probetable()
                    return 0
                num-=1
        self.reset_probetable()
        if num==0:
            return 1
        return 0
    '''
class Composed_partition(Partition):# set<set<int>>
    def __init__(self, base,addi):
        '''
        利用probe_table得到base+addi的partition
        Parameters
        ----------
        base-属性base的partition
        addi-属性addi的partition
        '''

        Partition.__init__(self,base,addi)

        if len(base)>len(addi):
            ba=addi
            ad=base
        else:
            ba=base
            ad=addi

        mapping= defaultdict(list)#dice<int ,intset>
        probe_table=Partition.probe_table
        i=1
        for equivalences in ba:#set<set<int>>
            for e in equivalences:
                probe_table[e] = i
            #probe_table[next(iter(equivalences))] = i
            mapping[i]=[]
            i+=1
        for equivalences in ad:
            it=iter(equivalences)
            for x in it:
                if probe_table[x]!=-1:
                    old=mapping.get(probe_table[x])
                    old.append(x)
            it = iter(equivalences)
            for x in it:
                s= mapping.get(probe_table[x])
                if s!=None and len(s)>1:
                    self.add(frozenset(s))
                mapping[probe_table[x]]=[]
        i=1
        for equivalences in ba:
            for e in equivalences:
                probe_table[e]=-1
            #probe_table[next(iter(equivalences))] = -1
    '''
    @classmethod
    def build_s(self,l_partitions):#list of partitions
        joined=[]
        if l_partitions.size()>1:
            result=l_partitions.get(0)
            for i in range(l_partitions.size()):
                result=Composed_partition(result, l_partitions.get(i))
                joined.append(result)
        elif l_partitions.size()==1:
            result = l_partitions.get(0)
        return joined
    '''
    @classmethod
    def build_p(self, l_partitions):  # list of partitions
        result = l_partitions[0]
        if len(l_partitions) > 1:
            for i in range(len(l_partitions)):
                if i==0:continue
                if len(result)==0:return result
                result = Composed_partition(result, l_partitions[i])
        return result#partition

class MM_partitions(dict):#dict<cardi,dict<colle,partition>>
    PARTITION_THRESHOLD = 10000
    def __init__(self, num_cols):
        super().__init__()
        self.num_cols=num_cols
        self.key=attr_combination(num_cols)
        self.usage_counter={}#dict<colle,int>
        self.lrh_partitions=[]#list<colle>
        self.total_count={}#dict<colle,int>
        for i in range(self.num_cols):
            self[i]={}
    def get_total(self):
        count=0
        for key in self.total_count.keys():
            count+=self.total_count.get(key)
        return count
    def get_p(self,colle):#由colle直接获得partition
        result=self.get(colle.cardi()).get(colle)
        if result!=None:
            if colle in self.lrh_partitions:self.lrh_partitions.remove(colle)
            self.lrh_partitions.append(colle)
            self.free_space()
        return result
    def get_count(self):
        c=0
        for ele in self.values():
            c+=len(ele)
        return c
    def free_space(self):
        if self.get_count()>MM_partitions.PARTITION_THRESHOLD+self.num_cols:
            usage_c=list(self.usage_counter.values())
            usage_c.sort()
            median=usage_c[int(len(usage_c)/2)]
            if len(usage_c)%2==0:
                median+=usage_c[int(len(usage_c)/2+1)]
                median/=2
            num_del=(MM_partitions.PARTITION_THRESHOLD+self.num_cols)/2
            deleted=0
            it=iter(self.lrh_partitions)
            for i in it:
                if deleted<num_del:
                    if not i.isatomic() and self.usage_counter.get(i)<=median:
                        self.lrh_partitions.remove(i)
                        self.remove_partition(i)
                        self.usage_counter.pop(i)
                        deleted+=1

    def add_partition(self,parti):#partition
        cardi=parti.indices.cardi()
        self.get(cardi)[parti.indices]=parti
        self.lrh_partitions.append(parti.indices)
        self.usage_counter[parti.indices]=1
        self.total_count[parti.indices]=1
    def add_partitions(self,partitions):
        for i in partitions:
            self.add_partition(i)
    def remove_partition(self,key):#colle
        cardi=key.cardi()
        self.get(cardi).remove(key)
    def get_matching(self,colle):
        '''
        得到用于生成colle的partition的合适的子集partitions列表
        Parameters
        ----------
        colle--node的indices

        Returns--list<partition>
        -------

        '''
        cop=copy.deepcopy(colle)
        matching_partitions=[]
        n_covered=cop.cardi()
        last_match=n_covered

        while n_covered>0:
            flag = 1
            #we don't need to check the sizes above the last match size again
            c_cardi=min(n_covered,last_match)
            while c_cardi>0 and flag==1:
                candidates=self.get(c_cardi)#项数为c_cardi的dict<colle,partition>
                for candi in candidates.keys():
                    if candi.is_subset(cop):
                        matching_partitions.append(self.get_p(candi))
                        n_covered-=c_cardi
                        cop=cop.union(candi)#copy.remmove(candi)
                        last_match=c_cardi
                        flag=0#回到外层循环
                        break
                c_cardi-=1
        return matching_partitions

class Dependencies(dict):#dict<colle,set<colle>> 单个属性->包含此属性的左手边
    SPLIT_THRESHOLD=1000
    def __init__(self, lhs):
        '''

        Parameters
        ----------
        lhs--list<int> 用于构建搜索空间的属性索引
        '''
        super().__init__()
        self.lhs=lhs
        for i in lhs:
            colle=int_to_indices(i)
            self[colle]=[]

    def inferred(self,colle):
        '''

        Parameters
        ----------
        colle--待推断节点的indices

        Returns--1 or 0
        -------

        '''
        for attr in self.keys():
            if attr.is_subset(colle):
                for dep in self.get(attr):
                    if colle.is_superset(dep):
                        return 1
        return 0
    def add(self,colle):
        for attr in self.keys():
            flag=1
            if attr.is_subset(colle):
                deps=self.get(attr)
                for dep in deps[:]:
                    if colle.is_subset(dep):
                        deps.remove(dep)
                    if colle.is_superset(dep):
                        flag=0#直接回到外层循环
                        break
                if flag:deps.append(colle)
        self.rebalance()
    def pruned_subsets(self,subsets):#list<colle>
        pruned=[]
        for sub in subsets:
            if self.inferred(sub):
                pruned.append(sub)
        return pruned#list<colle>
    def rebalance(self):#3.5
        '''
        we rebalance those data structures after adding new combinations by creating sub-lists for column pairs.
        '''
        flag=-1
        while flag!=0:
            flag=0
            attrs = self.keys()
            for key in attrs:
                if len(self.get(key)) > self.SPLIT_THRESHOLD:
                    self.rebalance_list(key)
                    flag=1
    def rebalance_list(self,key):#colle
        deps=self.get(key)
        cols=key.union(list_to_indices(self.lhs))
        for i in cols.get_bits():
            newkey=key.set_copy(i,1)
            newlist=[]
            self.setdefault(newkey,newlist)
            for dep in deps:
                if newkey.in_subset(dep):
                    self[newkey].append(dep)
        self.pop(key)

class Non_dependencies(dict):#dict<colle,set<colle>> 单个属性->包含此属性的左手边
    SPLIT_THRESHOLD = 1000
    def __init__(self, lhs):
        '''

        Parameters
        ----------
        lhs-list<int> 用于构建搜索空间的属性索引
        '''
        super().__init__()
        self.lhs=lhs
        for i in lhs:
            colle=int_to_indices(i)
            self[colle]=[]
    def inferred(self,colle):
        '''

        Parameters
        ----------
        colle--待推断节点的indices

        Returns--1 or 0
        -------

        '''
        for attr in self.keys():
            if attr.is_subset(colle):
                for dep in self.get(attr):
                    if colle.is_subset(dep):
                        return 1
        return 0
    def add(self,colle):
        for attr in self.keys():
            flag=1
            if attr.is_subset(colle):
                ndeps=self.get(attr)
                for ndep in ndeps[:]:
                    if colle.is_subset(ndep):
                        flag=0#需要直接回到外层循环
                        break
                    if colle.is_superset(ndep):
                        ndeps.remove(ndep)
                if flag:ndeps.append(colle)
        self.rebalance()
    def pruned_supersets(self,supersets):#list<colle>
        pruned=[]
        for sup in supersets:
            if self.inferred(sup):
                pruned.append(sup)
        return pruned#list<colle>
    def rebalance(self):#3.5
        '''
        we rebalance those data structures after adding new combinations by creating sub-lists for column pairs.
        '''
        flag=-1
        while flag!=0:
            flag=0
            attrs = self.keys()
            for key in attrs:
                if len(self.get(key)) > self.SPLIT_THRESHOLD:
                    self.rebalance_list(key)
                    flag=1
    def rebalance_list(self,key):#colle
        deps=self.get(key)
        cols=key.union(list_to_indices(self.lhs))
        for i in cols.get_bits():
            newkey=key.set_copy(i,1)
            newlist=[]
            self.setdefault(newkey,newlist)
            for dep in deps:
                if newkey.in_subset(dep):
                    self[newkey].append(dep)
        self.pop(key)

if __name__ == "__main__":
    dfd= DFD(sys.argv[1])#计算好的剥离分区
    t0 = time.time()
    dfd.run()
    print("\t=> Execution Time: {} seconds".format(time.time() - t0))
    print("\t=> {} Rules Found".format(dfd.count()))
    for k,i in dfd.FD.items():
        for l in i:
            print(colle_to_attr(l) +"->"+ attrset[k])

