from FDtree import FDTree
from pandas import *
import time
import copy
def violated(a,b,invalid_fds):
    global listofcolumns
    global atts_num
    dis = [] #值不同的属性的序号
    arg = [] #值相同的属性的序号
    for x in range(0,atts_num):
        if a[x] == b[x]:
            arg.append(x)
        else:
            dis.append(x)

    for a in dis:
        rhs = []
        rhs.append(a)
        invalid_fds.add(arg,rhs) #不需要考虑是否存在重复，因为add本身就不会重复添加，不过是否要考虑存在general?

#解决NCOVER冗余的问题
def Ncover_refine(fdtree):
    fds = list(fdtree.read_fds())
    for fd in fds:
        for rhs in fd[1]: #提取出单个的右边属性，因为get general函数需要
            general_lhss =  list(fdtree.get_fd_and_generals(list(fd[0]),rhs))
            if fd[0] in general_lhss : general_lhss.remove(fd[0])
            if(general_lhss):
                for lhs in general_lhss:
                    fdtree.remove(lhs,rhs)


#NCOVER和最后要输出DEPS不同，DEPS里面要的是最小函数依赖，如果新加的里面有更细分的，就舍去，而NCOVER要的是最细分的依赖，如果加的时候存在更宽泛的fd，需要删去general的
def negative_cover(relation):
    global atts_num
    NCOVER = FDTree(atts_num)#初始化一棵invalid_fdtree，当前只有[]->all attributes
    for i in range(0,len(relation)):
        for j in range(i+1,len(relation)):
            violated(relation[i],relation[j],NCOVER)
    Ncover_refine(NCOVER)
    return NCOVER

#初始化一棵[]->[all attributes]的fdtree
def initialise():
    global atts_num
    deps = FDTree(atts_num)
    deps.add([],range(atts_num))
    return deps




#凑新的lhs是在D_lhs上只加一个属性，并且加完之后还是要检查是否已经不在entail ND了
def specialise(fdtree, D , ND_lhs):#是根据D的lhs来加一个属性，不是ND的lhs
    att_num = 4
    for i in range(att_num):
        if ((i not in ND_lhs) and (i != D[1])):
            lhs = set(D[0][:])  # 把ND的lhs提出来
            lhs.add(i)  # 加一个新属性构成新的lhs，并且用set保证升序
            rhs = []
            rhs.append(D[1])
            if not fdtree.fd_has_generals(list(lhs),rhs[0]):
                fdtree.add(list(lhs), rhs)  # 不会entail ND、新的D，加入到Q中


def bottom_up(r):
    global listofcolumns
    DEPS = initialise() #初始化
    NCOVER = negative_cover(r) #ncover是一个list，里面的invalid_fd用tuple格式的({lhs},[rhs])存
    for ND in list(NCOVER.read_fds()): #ND是({lhs},[rhss])
        lhs = ND[0]
        rhss = ND[1]
        for rhs in rhss:#rhss要拆开，因为get general函数要求右手边是单个属性
           tmp = copy.deepcopy(DEPS) #拷贝一个对象
           D_lhss = list(DEPS.get_fd_and_generals(list(lhs),rhs)) #找到所有能entail ND的D，返回的是一个list，里面都是set格式的lhs
           for D in D_lhss:
               tmp.remove(D,rhs)
               specialise(tmp,(list(D),rhs),lhs)
               DEPS = copy.deepcopy(tmp)
    return DEPS

def index_to_name(res):
    global listofcolumns
    result = []
    for fd in res:
       lhs_n = []
       for att in fd[0]:
           lhs_n.append(listofcolumns[att])
       for rhs in fd[1]:
           result.append((lhs_n,listofcolumns[rhs]))
    return result



infile = open('abalone.csv')
data = read_csv(infile)
ntuples = len(data.index) #元组数量
listofcolumns = data.columns.tolist() #属性列表,是index格式,再转list
atts_num = len(listofcolumns) #属性个数
print("ncover:",list(negative_cover(data.values).read_fds()))
t0 = time.time()
result = bottom_up(data.values)
result = list(result.read_fds())#转成list，此时属性都还是序号
result = index_to_name(result)
print("num of fds:",result.__len__())
print(result)
print(format(time.time() - t0))
#print_result(result)
