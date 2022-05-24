import numpy as np


def GetSublt(Pa, array, val, layer, l):  # Pa为断开后的引领树索引 array即AL val为引领树中心节点 l为层数
    ind = [i for i, x in enumerate(Pa) if x == val]  # 指向当前节点的索引
    layer[ind] = l
    for i in range(len([i for i, x in enumerate(Pa) if x == val])):  # 查看Pa中有几个指向val的节点，对应引领树的分支
        array = np.append(array, ind[i])  # 将指向当前节点的节点索引加入
        if ind[i] not in Pa:
            pass
            # print('结束此分支')
        else:
            array = GetSublt(Pa, array, ind[i], layer, l+1)  # 层数+1
    return array


def EuclidianDist2(X1, X2):
    ###Using broadcasting, simpler and faster!
    tempM = np.sum(X1 ** 2, 1).reshape(-1, 1)  ##行数不知道，只知道列数为1
    tempN = np.sum(X2 ** 2, 1)  # X2 ** 2: element-wise square, sum(_,1): 沿行方向相加，但最后是得到行向量
    sqdist = tempM + tempN - 2 * np.dot(X1, X2.T)
    sqdist[sqdist < 0] = 0
    return np.sqrt(sqdist)


def GetNeighbor(Pa, array, val):
    array = np.append(array, val)  # 添加自身为邻域
    # if Pa[val] >= 0:
    #     array = np.append(array, Pa[val])  # 添加父节点为邻域
    ind = [i for i, x in enumerate(Pa) if x == val]
    for i in range(len(ind)):  # 添加子节点为邻域
        array = np.append(array, ind[i])
    return array


class LeadingTree:
    def __init__(self, X_train, y_train, dc, lt_num):
        self.label_num = len(np.unique(y_train))
        self.X_train = X_train
        self.dc = dc
        self.lt_num = lt_num
        # print('*' * 30 + str('\n计算距离矩阵D'))
        self.D = EuclidianDist2(X_train, X_train)

        # print('*' * 30 + str('\n计算样本局部密度'))
        tempMat1 = np.exp(-(self.D ** 2))
        tempMat = np.power(tempMat1, self.dc ** (-2))
        self.density = np.sum(tempMat, 1) - 1

        # print('*' * 30 + str('\n将向量按降序排序，得到新的下标向量'))
        Q = np.argsort(self.density)[::-1]

        # print('*' * 30 + str('\n计算通往密度更高的最近数据点的距离δ以及父节点Pa'))
        self.delta = np.zeros(len(X_train))
        self.Pa = np.zeros(len(X_train), dtype=int)
        for i in range(len(X_train)):
            # print('正在计算第%d个点' % i)
            if i == 0:
                self.delta[Q[i]] = max(self.D[Q[i]])
                self.Pa[Q[i]] = -2
            else:
                # 删除密度较小的列
                greaterInds = Q[0:i]
                D_A = self.D[Q[i], greaterInds]
                # 从局部密度大于它的那些数据点中，选出离它距离最近的那个数据点对应的距离
                self.delta[Q[i]] = min(D_A)
                self.Pa[Q[i]] = greaterInds[np.argmin(D_A)]

        # print('*' * 30 + str('\n计算局部密度与距离的乘积，得到各个点为中心点的可能性γ'))
        gamma = np.zeros(len(X_train))
        for i in range(len(X_train)):
            gamma[i] = self.density[i] * self.delta[i]
        gamma_D = np.argsort(gamma)[::-1]

        # print('*' * 30 + str('\n断开引领树'))
        for i in range(self.lt_num):
            self.Pa[gamma_D[i]] = -2

        # print('*' * 30 + str('\n提取lt_num个子树'))
        self.layer = np.zeros(len(self.Pa), dtype=int)
        self.AL = [np.zeros((0, 1), dtype=int) for i in range(lt_num)]  # AL[i]为某颗子树的所有节点的索引
        for i in range(self.lt_num):
            self.AL[i] = np.append(self.AL[i], gamma_D[i])  # 传入引领树中心节点
            self.AL[i] = GetSublt(self.Pa, self.AL[i], gamma_D[i], self.layer, 1)  # 根据中心节点找出引领树子树

        self.layer = self.layer + 1

        edgesO = np.array(list(zip(range(len(self.Pa)), self.Pa)))
        ind = edgesO[:, 1] > -2
        self.edges = edgesO[ind,]
