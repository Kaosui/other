'''
Kmeans:
指定k个聚类中心, 遍历数据集, 将每一个数据点指派到最近的类.
如果两次迭代后类的元素没有发生变化, 算法停止

实现中, 我们维护一个数组arr, 里面存放k个集合set用来存储各个类的元素下标
每一次迭代用 new_arr 存放本次结果, 如果 new_arr == arr 算法停止, 否则令 arr = new_arr
'''
import numpy as np
import random

class Kmeans(object):
    def __init__(self, k):
        '''k是类的个数'''
        self.k = k
        self.center = np.zeros(k) #k个中心
        self.arr = np.array([set() for _ in range(k)]) #k个集合
    def dist(self, x1, x2):
        #计算两个点的欧氏距离的平方
        return np.linalg.norm(x1-x2, ord=2)
    def send_to_nearest(self, X, i, arr):
        '''将数据点X[i]送到arr中对应的集合
        X: 输入数据
        i: 添加元素的索引
        '''
        index, min_dist = -1, float('inf')
        for j in range(self.k):
            if self.dist(X[i], self.center[j]) < min_dist:
                min_dist = self.dist(X[i], self.center[j])
                index = j
        arr[index].add(i)
    def train(self, X):
        #输入数据集X
        n,m = X.shape  #样本数, 样本维度
        #随机选取k个点作为聚类中心
        self.center = np.resize(self.center, (self.k,m)) 
        index = list(range(n))
        np.random.shuffle(index)
        for i in range(self.k):
            self.center[i] = X[index[i]]
        #遍历数据集, 将数据点发配到最近的聚类中心, 至收敛
        while True:
            #new_arr = self.arr
            new_arr = np.array([set() for _ in range(self.k)])
            for i in range(n):
                self.send_to_nearest(X, i, new_arr)
            #更新聚类中心
            #print(new_arr)
            for i in range(self.k):
                length = len(new_arr[i])
                s = 0
                for j in new_arr[i]: s+=X[j]
                self.center[i] = s/length
            if (new_arr == self.arr).all(): break
            self.arr = new_arr
    def get_result(self, X):
        #返回各类的数据及类中心
        result = [[] for _ in range(self.k)]
        for i in range(self.k):
            for j in self.arr[i]:
                result[i].append(X[j])
        return result, self.center



if __name__ == "__main__":
    X = np.random.randn(1000,2) #生成2维空间的1000个随机样本点

    kmeans = Kmeans(3) #分为3类
    kmeans.train(X)
    
    result, center = kmeans.get_result(X)

    #可视化
    x1, x2, y1, y2, x3, y3 = [],[],[],[],[],[]
    center_x, center_y = [],[]
    for i in result[0]:
        x1.append(i[0])
        y1.append(i[1])

    for i in result[1]:
        x2.append(i[0])
        y2.append(i[1])

    for i in result[2]:
        x3.append(i[0])
        y3.append(i[1])

    for i in center:
        center_x.append(i[0])
        center_y.append(i[1])

    from matplotlib import pyplot as plt

    plt.scatter(x1,y1,color='r')
    plt.scatter(x2,y2,color='b')
    plt.scatter(x3,y3,color='g')
    plt.scatter(center_x, center_y, marker='x', color='c') # 'x'表示类中心

    plt.show()