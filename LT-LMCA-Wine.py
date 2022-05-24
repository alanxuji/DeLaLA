from collections import Counter
from matplotlib import pyplot as plt
import LeadingTree as lt
import numpy as np
from sklearn import datasets
import LMCA_Mine as lm
from sklearn.preprocessing import MinMaxScaler
import datetime


def EuclidianDist2(X1, X2):
    ###Using broadcasting, simpler and faster!
    tempM = np.sum(X1 ** 2, 1).reshape(-1, 1)  ##行数不知道，只知道列数为1
    tempN = np.sum(X2 ** 2, 1)  # X2 ** 2: element-wise square, sum(_,1): 沿行方向相加，但最后是得到行向量
    sqdist = tempM + tempN - 2 * np.dot(X1, X2.T)
    sqdist[sqdist < 0] = 0
    return np.sqrt(sqdist)


def EuclidianDistsq(X1, X2):
    ###Using broadcasting, simpler and faster!
    tempM = np.sum(X1 ** 2, 1).reshape(-1, 1)  ##行数不知道，只知道列数为1
    tempN = np.sum(X2 ** 2, 1)  # X2 ** 2: element-wise square, sum(_,1): 沿行方向相加，但最后是得到行向量
    sqdist = tempM + tempN - 2 * np.dot(X1, X2.T)
    sqdist[sqdist < 0] = 0
    return sqdist


start_t1 = datetime.datetime.now()
lt_num = 12  # 子树个数
wine = datasets.load_wine()
X = wine.data
y = wine.target
scalar = MinMaxScaler()
X = scalar.fit_transform(X)
lt1 = lt.LeadingTree(X_train=X, y_train=y, dc=0.19, lt_num=lt_num)  # 整个数据集构造引领树

X_train = np.zeros((0, 13))  # 初始化训练集为
y_train = np.zeros(0, dtype=int)  # 初始化训练集标签
index = np.zeros(0, dtype=int)  # 训练集对应的索引

for i in range(lt_num):  # 子树中心节点加入训练集
    index = np.append(index, lt1.AL[i][0])
    X_train = np.append(X_train, X[lt1.AL[i][0]].reshape(1, -1), axis=0)
    y_train = np.append(y_train, y[lt1.AL[i][0]])

max_layer = np.max(lt1.layer)
for i in range(len(X)):  # 最深层与次深层节点加入训练集
    if lt1.layer[i] == max_layer or lt1.layer[i] == max_layer - 1:
        index = np.append(index, i)
        X_train = np.append(X_train, X[i].reshape(1, -1), axis=0)
        y_train = np.append(y_train, y[i])

lmca = lm.LMCA(dimension=2, init_method="kpca", verbose=True, max_iter=100, stepsize=1.E-2,
               nn_active=False, length_scale=1.5, length_scale_test=1.5, k=2)  # 1.5~0.974
lmca.fit(X_train, y_train)

X_test = np.delete(X, index, axis=0)  # 去除训练集后得到测试集样本
y_test = np.delete(y, index, axis=0)  # 测试集标签
y_predict = np.zeros(len(y_test), dtype=int) - 1  # 预测标签

MatDist2 = EuclidianDistsq(X_train, X_test)  # 训练集和测试集对应的核矩阵
test_bnd_K = np.exp(-1 * lmca.length_scale_test * MatDist2).T
B = test_bnd_K.dot(lmca.Omega)  # 降维后的测试集
A = lmca.K.dot(lmca.Omega)  # 降维后的训练集

# 为每个测试样本找到欧氏距离最近的训练样本，预测二者标签相同。
D = EuclidianDist2(B, A)
Pa = np.zeros(len(y_predict), dtype=int)  # Pa[i]代表距离测试样本i最近的训练样本的索引
for i in range(len(y_predict)):
    index1 = np.argmin(D[i])
    Pa[i] = index1

y_predict = y_train[Pa]

arr = y_predict - y_test
count = Counter(arr)[0]
print("总准确率为", count / len(y_test))
end_t1 = datetime.datetime.now()
elapsed_sec = (end_t1 - start_t1).total_seconds()
print("共消耗: " + "{:.10f}".format(elapsed_sec) + " 秒")

label0 = 0
label1 = 0
label2 = 0

area = np.pi * 4 ** 2
colors = ['#00CED1', '#DC143C', '#000079', '#467500', '#613030', '#EA0000', '#84C1FF', '#8C8C00', '#FFFF37', '#A5A552',
          '#F00078', '#007979', '#00FFFF', '#FFD306', '#336666', '#FF00FF', '#02F78E', '#FF8000', '#5A5AAD', '#921AFF',
          '#6C3365', '#FF5809', '#28FF28', '#272727', '#D3D3D3', '#66CCFF']

plt.title("train_after", fontsize=22)  # 标题
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xmax=10, xmin=-10)
plt.ylim(ymax=10, ymin=-10)
for i in range(len(X_train)):
    if y_train[i] == 0:
        label0 = label0 + 1
        plt.scatter(A[i, 0], A[i, 1], s=area, c=colors[0], marker='o', alpha=0.4, label='类别0')
    if y_train[i] == 1:
        label1 = label1 + 1
        plt.scatter(A[i, 0], A[i, 1], s=area, c=colors[1], marker='^', alpha=0.4, label='类别1')
    if y_train[i] == 2:
        label2 = label2 + 1
        plt.scatter(A[i, 0], A[i, 1], s=area, c=colors[2], marker='s', alpha=0.4, label='类别2')
    pass
plt.show()

plt.title("test_after", fontsize=22)  # 标题
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xmax=10, xmin=-10)
plt.ylim(ymax=10, ymin=-10)
for i in range(len(X_test)):
    if y_test[i] == 0:
        plt.scatter(B[i, 0], B[i, 1], s=area, c=colors[0], marker='o', alpha=0.4, label='类别0')
    if y_test[i] == 1:
        plt.scatter(B[i, 0], B[i, 1], s=area, c=colors[1], marker='^', alpha=0.4, label='类别1')
    if y_test[i] == 2:
        plt.scatter(B[i, 0], B[i, 1], s=area, c=colors[2], marker='s', alpha=0.4, label='类别2')
plt.show()
