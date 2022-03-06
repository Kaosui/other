'''
存储前向传播的参数，并通过反向传播更新参数
ai是第i层的输出，  hi是ai经激活函数后的输出

输入格式：
X: [feature_num, batch]
Y: [nums_class, batch]  经OneHot编码的标签集

MLP接收参数：
layer -> list[int]: [l1, l2, ..., ln], 其中li表示第i层的神经元数
activation -> list[func] : [act1, act2, ...,actn], 其中act是定义在function.py中的激活函数, acti表示第i层的激活函数
lossfunction -> func : 定义在function.py中的损失函数

参数：
forward_W  存储矩阵
forward_b  存储偏置
forward_a  存储输出
forward_h  存储激活后的输出

backward_h 对h的梯度
backward_a 对a的梯度
backward_W 对W的梯度
backward_b 对b的梯度
'''
from functions import *
from derivative import *

class MLP(object):
    def __init__(self, layer, activation, lossfunction):
        '''
        layer: 输入数组，从第一个隐藏状态开始到输出层的神经元数量
               [n1,n2,...nN]
        activation: 各层的激活函数, 与layer对应
        '''
        #前向传播参数
        self.forward_a = []
        self.forward_h = []
        self.forward_W = []
        self.forward_b = []
        #反向传播参数
        self.backward_a = []
        self.backward_h = []
        self.backward_W = []
        self.backward_b = []
        #各层神经元数
        self.layer = layer
        self.activation = activation
        self.lossfunction = lossfunction
    
    def get_parameters(self):
        return self.forward_W, self.forward_b, self.forward_a, self.forward_h

    def initialize(self,X):
        '''
        根据输入X：n×N (特征 × 样本数), 初始化参数并加入相应的list中
        Wi : ni × n(i-1)   i>=0,  n(-1)=n
        bi : ni × 1
        ai : ni × N
        hi : ni × N
        '''
        n,N = X.shape
        self.forward_W.append( np.random.normal(0,np.sqrt(2/self.layer[0]),(self.layer[0],n)) )
        self.forward_b.append( np.zeros((self.layer[0],1)) )
        self.forward_a.append( np.matmul(self.forward_W[0],X)+self.forward_b[0] )
        self.forward_h.append( self.activation[0](self.forward_a[0]) )
        
        for i in range(1,len(self.layer)):
            self.forward_W.append( np.random.normal(0,np.sqrt(2/self.layer[i]),(self.layer[i],self.layer[i-1])) )
            self.forward_b.append(np.zeros((self.layer[i],1)))
        #前向传播计算
            self.forward_a.append( np.matmul(self.forward_W[i],self.forward_h[i-1])+self.forward_b[i] )
            self.forward_h.append( self.activation[i](self.forward_a[i]) )

        #反向传播参数list初始化
        for i in range(len(self.layer)):
            m,n = self.forward_a[i].shape
            self.backward_a.append(np.zeros((m,n)))
            q,w = self.forward_b[i].shape
            self.backward_b.append(np.zeros((q,w)))
            e,r = self.forward_W[i].shape
            self.backward_W.append(np.zeros((e,r)))
            z,x = self.forward_h[i].shape
            self.backward_h.append(np.zeros((z,x)))

    def forward(self,X):
        '''
        与initialize相同, 前向更新各层输出a和h
        '''
        self.forward_a[0] = np.matmul(self.forward_W[0],X)+self.forward_b[0]
        self.forward_h[0] = self.activation[0](self.forward_a[0])

        for i in range(1,len(self.layer)):
            self.forward_a[i] = np.matmul(self.forward_W[i],self.forward_h[i-1])+self.forward_b[i]
            self.forward_h[i] = self.activation[i](self.forward_a[i])

    def loss(self,X,Y,dim=1):
        ''' 
        标签集Y, 返回损失函数的值
        X: n×N   n是特征数, N是样本数
        Y: 离散标签的OneHot编码   输出层维数 × N
        lossfunction: 交叉熵  MSE  MAE
        '''
        self.forward(X)
        return self.lossfunction(self.forward_h[-1], Y,dim=dim)


    def backward(self,X,Y,lr):
        '''
        反向传播参数更新, 计算完梯度之后更新forward_W, forward_b
        lr: 学习率
        '''
        self.forward(X)
        Random = np.random.randint(X.shape[1])  #SGD随机采样的样本
        if self.lossfunction == CrossEntropy or self.lossfunction == MSE:
            d0 = (self.forward_h[-1][:,Random]-Y[:,Random]).reshape(-1,1)
        #此处应有补充其他损失函数的情况
        #补充：MSE和交叉熵在输出层求导的结果是一致的
        self.backward_a[-1] = d0
        self.backward_W[-1] = np.matmul( self.backward_a[-1], self.forward_h[-2][:,Random].reshape(1,-1) )
        self.backward_b[-1] = self.backward_a[-1]
        for i in range(len(self.layer)-2,0,-1):
            self.backward_h[i] = np.matmul( self.forward_W[i+1].T, self.backward_a[i+1] )
            self.backward_a[i] = self.backward_h[i] * derivative(self.activation[i], self.forward_a[i][:,Random].reshape(-1,1))
            self.backward_b[i] = self.backward_a[i]
            self.backward_W[i] = np.matmul( self.backward_a[i], self.forward_h[i-1][:,Random].reshape(1,-1) )

        self.backward_h[0] = np.matmul( self.forward_W[1].T, self.backward_a[1] )
        self.backward_a[0] = self.backward_h[0] * derivative(self.activation[0], self.forward_a[0][:,Random].reshape(-1,1))
        self.backward_b[0] = self.backward_a[0]
        self.backward_W[0] = np.matmul( self.backward_a[0], X[:,Random].reshape(1,-1) )
        #更新参数
        for i in range(len(self.layer)):
            self.forward_W[i] -= lr*self.backward_W[i]
            self.forward_b[i] -= lr*self.backward_b[i]

    def score(self,X,Y):
        '''
        返回预测准确率
        '''
        self.forward(X)
        W,b,a,h = self.get_parameters()
        y_hat = h[-1]
        correct = 0
        correct += sum(np.argmax(y_hat,0)==np.argmax(Y,0))
        accuracy = correct/X.shape[1]
        return accuracy

    def predict(self, X):
        '''
        返回集合X的预测概率值
        '''
        self.forward(X)
        W,b,a,h = self.get_parameters()
        return np.argmax(h[-1],0).astype(np.int32)

    def predict_proba(self, X):
        '''
        返回预测概率
        '''
        self.forward(X)
        W,b,a,h = self.get_parameters()
        return h[-1]


if __name__=='__main__':

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    
    #数据取自《机器学习》周志华, 西瓜数据集
    X=np.array([
        (0.679,0.460),
        (0.774,0.376),
        (0.634,0.264),
        (0.608,0.318),
        (0.556,0.215),
        (0.403,0.237),
        (0.481,0.149),
        (0.437,0.211),
        (0.666,0.091),
        (0.243,0.267),
        (0.245,0.057),
        (0.343,0.099),
        (0.639,0.161),
        (0.657,0.198),
        (0.360,0.370),
        (0.593,0.042),
        (0.719,0.103)
    ])
    Y=np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])

    #转为输入格式以及标签OneHot转换
    y = Y.reshape(-1,1)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray().T
    X = X.T

    #定义隐层维度分别为100,20,2
    layer = [100,20,2]
    activation = [relu,relu,softmax]

    #定义分类器
    clf = MLP(layer, activation, CrossEntropy)
    clf.initialize(X)

    #训练
    for i in range(1000):
        clf.forward(X)
        clf.backward(X,y,lr=5e-4)
        if((i+1)%100==0):
            print('epoch:%d, loss:%f'%(i+1,clf.loss(X,y)))

    print('准确率：', clf.score(X, y))
    print('实际结果：', Y)
    print('预测结果：', clf.predict(X))
    print('预测概率：', clf.predict_proba(X))