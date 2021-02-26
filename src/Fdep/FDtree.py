'''
[1] Papenbrock et al - A Hybrid Approach to Functional Dependency Discovery (2016)
'''
import logging

logger = logging.getLogger(__name__)


class FDCollection(object):
    def __init__(self, n_atts):
        self.n_atts = n_atts

    def add(self, lhs, rhss):
        raise NotImplementedError

    def l_close(self, pat):
        raise NotImplementedError

    @property
    def n_fds(self):
        raise NotImplementedError

    def read_fds(self):
        raise NotImplementedError


class FDNode(object):
    def __init__(self, att=-1, n_atts=0):
        self.att = att
        # self.idx = [True]*n_atts
        self.link = {}  # 记录孩子节点的dic，键是孩子节点的属性序号，值是孩子节点
        self.parent = None
        self._rhs = [False] * n_atts
        self.active = False

    def set_rhss(self, rhss):
        for i in rhss:
            self._rhs[i] = True
        self.active = True  # 表示有右手边？

    def get_children(self):
        for i in sorted(self.link.keys()):
            yield self.link[i]

    def invalidate(self, invalid_rhss):
        for i in invalid_rhss:
            self._rhs[i] = False

    def __repr__(self):
        return str("<FDNode>{}=>{}".format(self.get_lhs(), str(self.get_rhss())))

    def get_rhss(self):
        return [i for i, j in enumerate(self._rhs) if j]  # 值为True的右手边属性的序号会被打印出来

    def get_lhs(self):
        base = set([])  # 给属性自动排序
        parent = self
        while parent is not None:  # 根节点的父母是Node，所以回溯到根节点的判定是parent=None
            if parent.att >= 0:
                base.add(parent.att)
            parent = parent.parent
        return base

    # 翻转右手边，作用？
    def flip(self):
        for i in range(len(self._rhs)):
            self._rhs[i] = not self._rhs[i]

    def add_child(self, child):
        # self.idx[child.att] = False
        self.link[child.att] = child
        child.parent = self

    def remove_rhs(self, rhs):
        self._rhs[rhs] = False
        if not any(self._rhs):  # any函数可以判断一个迭代器里面的值是否都是false，这里可以用来判断右手边是否为空
            self.active = False


class FDTree(FDCollection):
    '''
    Keeps a set of FDs stored in a tree.
    Implemented using descriptions found in [1]
    '''

    def __init__(self, n_atts=0):
        '''
        Initializes the object by setting the number of attributes
        contained in the functional dependencies to be stored.
        The tree only holds a reference to the root node.
        '''
        super(FDTree, self).__init__(n_atts)
        self.root = FDNode(n_atts=self.n_atts)
        self._n_fds = 0

    def _level_and_recurse(self, current_node, sought_depth, depth=0):
        '''
        Recursive function searching within the tree
        for all nodes at a given depth.
        Nodes do not store information on its depth
        so the depth is calculated along with the navigation
        by means of the depth parameter


        current_node -- FDNode, Current node in the navigation
        sought_depth -- int, Target depth
        depth -- int, current depth (default 0)
        '''
        if sought_depth == depth:
            yield current_node
        else:
            for att in sorted(current_node.link.keys()):
                for i in self._level_and_recurse(current_node.link[att], sought_depth,
                                                 depth + 1):  # for函数的每一次循环就相当于next来调用这个递归函数本身
                    yield i

    def get_level(self, sought_depth):
        '''
        Yields all nodes at a given depth by means

        sought_depth -- int, Target depth
        '''
        for i in self._level_and_recurse(self.root, sought_depth):
            yield i

    def _print_and_recurse(self, current_node, depth=1):
        '''
        Recursively print the nodes in the tree

        current_node -- FDNode, current node in the navigation
        depth -- int current depth
        '''
        print('\t' * depth, current_node.att, current_node._rhs, current_node.link)
        for i in sorted(current_node.link.keys()):
            self._print_and_recurse(current_node.link[i], depth + 1)

    def print_tree(self):
        '''
        Print all nodes in the tree
        '''
        self._print_and_recurse(self.root)

    def find_fd(self, lhs, rhs):
        '''
        Search in the FDTree for the FD lhs -> rhs
        lhs -- set with attribute ids in the left hand side
        rhs -- attribute id in the right hand side
        '''
        current_node = self.root
        s_lhs = sorted(lhs, reverse=True)  # 因为后面用pop弹出的是在末尾的，所以把原来的顺序做一个倒置
        while bool(s_lhs):
            next_att = s_lhs.pop()
            if current_node.link.get(next_att, False):  # 字典的get函数，用来返回指定属性的值，如果字典中不存在这个属性会返回False
                current_node = current_node.link[next_att]
            else:
                return False  # 说明不存在这个fd
        return current_node._rhs[rhs]  # 一直深入到左手边最后一个属性这个节点，最后检查它的右手边这个属性是否存在
        # 应该是_rhs？

    '''
    def first_fd(self,lhs,current_node):
        if current_node.active:
            for i,j in enumerate(current_node._rhs):
                if j==True:
                    current_node._rhs[i] = False #找个之后要把这个fd删除
                    if any(current_node._rhs): current_node.acitve=False #检查一下是否全部都变成了False
                    rhs = []
                    rhs.append(i)
                    return (lhs,rhs)
        else:
            for atts in current_node.link.keys():
                self.first_fd(lhs.append(atts) , current_node.link[atts])
    '''


    def _find_and_recurse(self, current_node, lhs):

        if current_node.active:
            yield current_node.get_rhss()

        if not bool(lhs) or not bool(current_node.link) or max(current_node.link.keys()) < lhs[
            -1]:  # lhs为空或当前节点没有子节点 ，子节点中没有比lhs第一个属性的序号更大的了
            return  # 提前结束，此路不同
        for ati, att in enumerate(lhs):
            next_node = current_node.link.get(att, False)
            if next_node:
                for fd in self._find_and_recurse(next_node, lhs[ati:]):
                    yield fd
        # for att in sorted(current_node.link.keys()):
        #     if att in lhs:
        #         next_node = current_node.link[att]
        #         for i in self._find_and_recurse(next_node, base.union([att]), lhs):
        #             yield i
        # elif att < lhs[-1]:
        #     break

    def find_rhss(self, lhs):
        '''
        Search in the FDTree for the FD lhs -> rhs
        lhs -- set with attribute ids in the left hand side
        rhs -- attribute id in the right hand side
        '''

        if len(lhs) == self.n_atts:  # 如果是左手边有所有属性，那么没意义
            return  # 返回什么呢？
        # print "LHS", lhs
        slhs = sorted(lhs, reverse=True)  # 这里需要翻转吗？？？
        for old_rhs in self._find_and_recurse(self.root, slhs):
            # print '\t\t',old_lhs, old_rhs
            yield old_rhs
        # print "--"
        # return False

    def add(self, lhs, rhss): #两个都是list
        """
        Adds a set of FDs to the tree of the form
        lhs -> rhs for each rhs in rhss

        lhs -- set of attribute ids in the left hand side
        rhss -- set of attribute ids in the right hand side
        """

        new_node = None
        current_node = self.root
        s_lhs = sorted(lhs, reverse=True)
        self._n_fds += len(rhss)

        while bool(s_lhs):
            next_att = s_lhs.pop()  # 弹出当前的第一个属性
            add = True  # ？？？
            if current_node.link.get(next_att, False):  # 如果子节点中有下一个属性，直接把当前节点更新为下个节点就行了
                current_node = current_node.link[next_att]
            else:
                new_node = FDNode(att=next_att, n_atts=self.n_atts)  # 如果子节点中没有这个属性的话 新建这个子节点
                current_node.add_child(new_node)  # 添加为当前节点的孩子
                current_node = new_node  # 更新当前节点
        current_node.set_rhss(rhss)

        return new_node  # 最终输出的这个节点上一定有lhs—>rhss这个fd
    '''
    此方法实践后发现不行，因为添加的时候顺序是随机的，只有在general——》special这个顺序此方法才是正确的
    def add_degeneral(self,lhs,rhs):
        
        # NCover 的fdtree在加入invalid fd时需要把它的general都删掉，这样才算non-redundant
        # rhs用的是单个属性，int
        new_node = None
        current_node = self.root
        s_lhs = sorted(lhs, reverse=True)
        self._n_fds += len(rhs)
        while bool(s_lhs):
            next_att = s_lhs.pop()  # 弹出当前的第一个属性
            if current_node.link.get(next_att, False):  # 如果子节点中有下一个属性，直接把当前节点更新为下个节点就行了
                current_node = current_node.link[next_att]
                if current_node._rhs[rhs[0]] and s_lhs: #后面这个条件是当这个属性是lhs的最后一个时，说明这个这个fd和要加入的一样，防止加一样的fd反而会删掉原来的
                    current_node._rhs[rhs[0]] =False #删掉路径上经过的，因为一定是general

            else:
                new_node = FDNode(att=next_att, n_atts=self.n_atts)  # 如果子节点中没有这个属性的话 新建这个子节点
                current_node.add_child(new_node)  # 添加为当前节点的孩子
                current_node = new_node  # 更新当前节点
        current_node.set_rhss(rhs)

        return new_node  # 最终输出的这个节点上一定有lhs—>rhss这个fd
    '''

    def _read_and_recurse(self, current_node, lhs):
        '''
        Recursively read all FDs in the FDTree

        current_node -- current node in the navigation
        lhs -- current left hand side
        '''
        if current_node.active:
            yield (lhs, current_node.get_rhss())

        for att in sorted(current_node.link.keys()):
            next_node = current_node.link[att]
            for fd in self._read_and_recurse(next_node, lhs.union([att])):  # union可以合并集合set
                yield fd

    def read_fds(self):  # 返回的是元组形式的fds
        '''
        Read all fds in the FDTree
        '''
        current_node = self.root
        base = set([])
        for i in self._read_and_recurse(current_node, base):
            yield i

    # def check_and_recurse(self, current_node, base, lhs, rhs):

    #     if current_node._rhs[rhs]:
    #         yield (base, rhs)
    #     for att in sorted(current_node.link.keys()):
    #         if att in lhs:
    #             next_node = current_node.link[att]
    #             for i in self.check_and_recurse(next_node, base.union([att]), lhs, rhs):
    #                 yield i
    #         elif att > max(lhs):
    #             break

    # 根据lhs这条路径上的任意节点的右手边是rhs，显然就是fd的general
    def _check_and_recurse(self, current_node, lhs, rhs):

        if current_node._rhs[rhs]:
            yield (current_node.get_lhs(), rhs)

        for ati, att in enumerate(lhs):
            next_node = current_node.link.get(att, False)
            if next_node:
                for fd in self._check_and_recurse(next_node, lhs[ati:], rhs):
                    yield fd

    # 要排除掉（lhs,rhs）本身这个fd吗 ##不用，NCOVER里如果有一样的也是需要删除然后细化
    def fd_has_generals(self, lhs, rhs):
        """
        rhs contains a single attribute
        """
        slhs = sorted(lhs)
        for old_lhs, old_rhs in self._check_and_recurse(self.root, slhs, rhs):
            return True  # 一次就够了，所以用return
        return False

    def get_fd_and_generals(self, lhs, rhs):
        slhs = sorted(lhs)
        for old_lhs, old_rhs in self._check_and_recurse(self.root, slhs, rhs):
            yield old_lhs

    def remove(self, lhs, rhs):
        '''
        Remove FD lhs->rhs from the FDTree

        '''
        self._n_fds -= 1  # fdtree的属性个数减一
        current_node = self.root
        s_lhs = sorted(lhs, reverse=True)
        while bool(s_lhs):
            next_att = s_lhs.pop()
            current_node = current_node.link.get(next_att, False)
            if not current_node:
                raise KeyError
        current_node.remove_rhs(rhs)  # 不用删掉这个节点，把这个节点的rhs置false就行

    def _specialize_and_recurse(self, current_node, lhs, rhss, pointer=0):

        # REMOVE
        for ati, att in enumerate(lhs[pointer:]):
            next_node = current_node.link.get(att, False)
            if next_node:
                for node in self._specialize_and_recurse(next_node, lhs, rhss, pointer + 1):
                    yield node
        # 上面的递归遍历到了头就会执行下面的，因为不存在next_node
        for rhs in rhss:
            if current_node._rhs[rhs]:  # 因为是按lhs里的属性遍历下来的，遇到的都是invalid fd的general，显然也是invalid，所以删去
                # REMOVE
                current_node.remove_rhs(rhs)
                self._n_fds -= 1
                invalid_lhs = current_node.get_lhs()
                # for new_att in (i for i in rhss if i != rhs):
                # 删掉一个invalid之后需要加一个新的更special的fd，并且这个fd不能是fdtree中已有fd的special
                for new_att in (i for i in range(self.n_atts) if i not in lhs and i != rhs):  # 找一个新的属性
                    new_lhs = invalid_lhs.union([new_att])  # 构成新的lhs，set形式？
                    if not self.fd_has_generals(new_lhs, rhs):  # 如果这个新的fd没有general，就可以加入到fdtree（保证最小依赖）
                        yield self.add(new_lhs, [rhs])  # add函数的返回值是新建节点

                    # new_lhs = invalid_lhs.union([new_att])
                    # if self.fd_has_generals(new_lhs, rhs):
                    #     continue
                    # yield self.add(new_lhs, [rhs])

    # specialize用的是invalid fd
    def specialize(self, lhs, rhss):
        slhs = sorted(lhs)
        out = list(self._specialize_and_recurse(self.root, slhs, rhss))  # 新生成的所有节点
        return out
    '''
    def l_close(self, pat):
        newpat = set(pat)

        while True:
            complement = reduce(set.union, [set([])] + [rhs for rhs in self.find_rhss(newpat)])
            if complement.issubset(newpat):
                break
            newpat.update(complement)
        return newpat
    '''
    @property
    def n_fds(self):
        return self._n_fds