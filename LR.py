'''
基于最速下降和牛顿法实现的二分类逻辑回归
输入数据维度
X：[batch, input_dim]
Y: [batch]

输出参数维度:
w: [input_dim, 1]
b: [1]

x·w + b = (x, 1)·(w,b)^T
输入矩阵扩充为 [batch_size, input_dim+1] , 在每一行的最后一维统一补上1
'''

class LogisticRegression(object):
    '''
    lr：学习率
    penalty：正则化项, 选择'l1'或'l2'或None
    max_iter：最大迭代次数
    C：正则化系数
    optim：优化算法, 'gd'为梯度下降, 'newton'为牛顿法
    '''
    def __init__(self, lr=0.1, penalty='l2', max_iter=100, C=1.0, optimizer='gd'):
        self.lr = lr
        self.penalty = penalty
        self.max_iter = max_iter
        self.C = C
        self.optim = optimizer
        self.w = np.zeros([1,1])  #模型参数 w

    def sigmoid(self, z):
        if z>=0: return 1/(1+np.exp(-z))
        else: return np.exp(z)/(1+np.exp(z))

    def fit(self, X, Y):
        '''
        X: [batch, input_dim]
        Y: [batch], 元素为0或1
        '''
        #将X在第0维上扩充一个数, 并赋值1
        batch, dim = X.shape
        self.w = np.resize(self.w, [dim+1,1])
        new_X = np.ones([batch, dim+1])
        for i in range(batch):
            for j in range(dim):
                new_X[i][j] = X[i][j]
        X = new_X
        Y = Y.reshape(-1,1)
        #训练参数
        if self.optim not in ['gd', 'newton']: raise ValueError('optim请从\'gd\'或者\'newton\'中选择')
        if self.optim == 'gd':
            self.fit_gd(X, Y)
        else:
            self.fit_newton(X, Y)    

    def fit_gd(self, X, Y):
        #计算惩罚项的梯度
        penalty_1 = 0  #惩罚项的一阶梯度
        if self.penalty == 'l2': penalty_1 = 2*self.C*self.w
        elif self.penalty == 'l1':
            dw_1 = self.w>0
            penalty_1 = self.C*dw_1
        #计算参数的梯度
        for step in range(self.max_iter):
            #计算预测值 Z = sigmoid(-X·w)   [batch, 1]
            Z = np.matmul(X,self.w)
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    Z[i][j] = self.sigmoid(Z[i][j])
            #计算梯度
            N,_ = X.shape
            dw = 0  #[input_dim+1, 1]
            for i in range(N):
                dw += (Z[i][0]-Y[i][0])*X[i].reshape(-1,1)
            dw = dw/N + penalty_1
            self.w -= self.lr*dw
            #print(self.loss(X[:,:-1],Y)[0])
            #print('%d complete'%(step+1))

    def fit_newton(self, X, Y):
        N, n = X.shape
        #计算惩罚项的一阶二阶梯度
        penalty_1 = 0  #惩罚项的一阶梯度
        penalty_2 = 0  #惩罚项的二阶梯度
        if self.penalty == 'l2':
            penalty_1 = 2*self.C*self.w #[dim+1, 1]
            penalty_2 = 2*self.C*np.eye(n) #[dim+1, dim+1]   np.eye()生成对角元为1的对角阵
        elif self.penalty == 'l1':
            penalty_1 = self.C*(self.w>0)
        for step in range(self.max_iter):
            #计算预测值 Z = sigmoid(-X·w)   [batch, 1]
            Z = np.matmul(X,self.w)
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    Z[i][j] = self.sigmoid(Z[i][j])
            #计算一阶梯度
            d1w = 0  #[input_dim+1, 1]
            for i in range(N):
                d1w += (Z[i][0]-Y[i][0])*X[i].reshape(-1,1)
            d1w  = d1w/N + penalty_1
            #计算二阶梯度
            d2w = 0
            for i in range(N):
                x = X[i].reshape(1,-1)
                d2w += Z[i][0]*(1-Z[i][0])*np.matmul(x.T, x) #[dim+1, dim+1]
            d2w = d2w/N + penalty_2
            #更新参数
            dw = -np.matmul( np.linalg.inv(d2w), d1w )
            self.w += dw
            #print('%d complete'%(step+1))

    def get_param(self):
        return self.w[:-1], self.w[-1]

    def score(self, X, Y):
        '''给出X和Y之间的预测准确率'''
        pre_Y = self.predict(X)
        return (pre_Y==Y).sum()/len(pre_Y)

    def predict(self, X):
        '''给出输入X的预测'''
        pre_Y = (np.matmul(X, self.w[:-1])+self.w[-1]).reshape(-1)  #[batch]
        for i in range(len(pre_Y)):
            pre_Y[i] = 1 if pre_Y[i]>0 else 0 #z>0的点对应sigomid(z)>0.5
        return pre_Y.astype(np.int32)

    def predict_proba(self, X):
        '''返回X的预测概率'''
        pre_Y = (np.matmul(X, self.w[:-1])+self.w[-1]).reshape(-1)
        for i in range(len(pre_Y)):
            pre_Y[i] = self.sigmoid(pre_Y[i])
        return pre_Y

    def loss(self, X, Y):
        '''返回损失函数的值'''
        pre_Y = self.predict_proba(X) #预测为正类的概率
        loss = 0
        for i in range(len(Y)):
            loss += -(Y[i]*np.log(pre_Y[i]) + (1-Y[i])*(np.log(1-pre_Y[i])))
        loss /= len(Y)
        return loss

if __name__ == "__main__":
    import numpy as np
    
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

    clf = LogisticRegression(max_iter=100, optimizer='newton')
    clf.fit(X, Y)

    w,b = clf.get_param()
    print('w:',w)
    print('b:',b)
    print('准确率：', clf.score(X, Y))
    print('实际结果：', Y)
    print('预测结果：', clf.predict(X))
    print('预测概率：', clf.predict_proba(X))