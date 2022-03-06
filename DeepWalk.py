import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import random

#读取数据, 这里先不考虑节点的特征X，单纯从邻接矩阵进行无监督学习，得到游走序列后用w2v进行训练，最后用Kmeans评估分类结果
A = np.fromfile(r'D:\coding\dataset\cora数据集\预处理\A.bin', dtype=np.int32).reshape(2708,2708)

#接下来用随机游走生成2*len(A)条序列供w2v训练，每一条序列限制长度为20
iter_num = 2 #每个节点迭代10次
seq_len = 20

#先定义一个类, 该类接收邻接矩阵A的一条边 w = A[i]，并由w产生跳转节点， LeetCode 528原题
class jump(object):
    def __init__(self, w):
        """
        :type w: List[int]
        """
        self.w = w
    def binary_search(self, s, r):
        #在前缀和数组s中查找满足s[i]<r<=s[i+1]的i
        left, right = 0, len(s)-1
        while(left<right):
            mid = (left+right)//2
            if(s[mid]>=r):
                #说明可能位置在左边或者就是该位置
                right = mid
            else:
                #说明可能位置在右边
                left = mid+1
        return right-1
    def pickIndex(self):
        """
        :rtype: int

        计算权值的前缀和s, s[i]表示前i个数的和, s[0]=0
        生成一个位于1~s[-1]之间的随机数r
        找到一个位置i, 满足s[i-1]<r<=s[i], 这里用二分查找
        于是此时选择的数即为第i个数

        复杂度分析：时间O(n), 空间O(n)
        """
        s = [0]*(len(self.w)+1)
        for i in range(len(self.w)): s[i+1] = s[i]+self.w[i]
        n = len(self.w)
        r = random.randint(1,s[-1])
        return self.binary_search(s, r)


#产生训练数据s
def gernerate_train_data(A, iter_num, seq_len):
    '''
    params:
        A:邻接矩阵
        iter_num:迭代次数
        seq_len:游走序列的长度

    return:
        句子列表 [sentence1, sentence2,...] shape:[len(A)*iter_num, seq_len]
    '''
    all_sentence = []
    for epoch in range(iter_num):
        for i in tqdm(range(len(A))):
            sentence = [] #存放当前游走的序列
            index = i #当前游走的节点下标
            for k in range(seq_len):
                sentence.append(str(index))
                J = jump(A[index])
                index = J.pickIndex() #下一个跳转的节点
            all_sentence.append(sentence)
    return all_sentence


all_sentence = gernerate_train_data(A, iter_num, seq_len) #生成训练数据


from gensim.models import Word2Vec
w2v = Word2Vec(
    sentences = all_sentence,
    size = 32,
    min_count = 0,
    workers = -1
)


#创建归一化向量字典
w2v_dict = {}
for i in range(len(A)):
    w2v_dict[i] = w2v.wv.__getitem__(str(i))/np.linalg.norm(w2v.wv.__getitem__(str(i)), ord=2)


#供Kmeans训练的数据
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 7)

X = np.array([
    w2v_dict[i] for i in range(len(A))
])

kmeans.fit(X)


#接下来评估DeepWalk+Kmeans的性能, 用nx作图并打印, 和原图进行比较
#为每一个节点指定颜色, 定义颜色编号
id2color = {
    0:'r',
    1:'g',
    2:'b',
    3:'m',
    4:'orange',
    5:'k',
    6:'c'
}

#需要得到图的边信息
edges = []
for i in range(len(A)):
    for j in range(len(A[i])):
        if A[i][j] != 0:
            edges.append([i,j])


all_nodes = np.array(list(range(len(A))))
#首先绘制原图
Y = np.fromfile(r'D:\coding\dataset\cora数据集\预处理\Y.bin', dtype=np.int32)
colormap1 = [id2color[i] for i in Y]
G1 = nx.Graph()
G1.add_nodes_from(all_nodes)
G1.add_edges_from(np.array(edges))

#绘制无监督学习的图
colormap2 = [id2color[i] for i in kmeans.labels_]
G2 = nx.Graph()
G2.add_nodes_from(all_nodes)
G2.add_edges_from(edges)


#画图
plt.subplot(1,2,1)
nx.draw(G1, with_labels=False, edge_color='y', node_color=colormap1, node_size=10)
plt.subplot(1,2,2)
nx.draw(G2, with_labels=False, edge_color='y', node_color=colormap2, node_size=10)
plt.show()
