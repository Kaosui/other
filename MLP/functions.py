import numpy as np

#sigmoid函数
def sigmoid(input):
    X = input.astype('float64')
    def sig(x):
        return 1/(1+np.exp(-x)) if x>0 else np.exp(x)/(1+np.exp(x))
    if len(X.shape)==1:
        for i in range(len(X)):
            X[i] = sig(X[i])
        return X
    else: 
        for i in range(len(X)):
            X[i] = sigmoid(X[i])
        return X
        

#softmax函数
def softmax(input, dim=1):
    ''' dim=0对行, dim=1对列'''
    X = input.astype('float64')
    if dim == 1:
        return softmax(X.T,dim=0).T
    if len(X.shape)==1:
        temp = np.max(X)
        X -= temp
        X = np.exp(X)
        temp = sum(X)
        X/=temp
        return X
    else:
        for i in range(len(X)):
            X[i] = softmax(X[i])
        return X

#relu函数
def relu(input):
    X = input.astype('float64')
    if len(X.shape)==1:
        for i in range(len(X)):
            X[i] = max(X[i],0)
        return X
    else:
        for i in range(len(X)):
            X[i] = relu(X[i])
        return X

#tanh函数
def tanh(input):
    X = input.astype('float64')
    if len(X.shape)==1:
        for i in range(len(X)):
            X[i] = np.tanh(X[i])
        return X
    else:
        for i in range(len(X)):
            X[i] = tanh(X[i])
        return X

#交叉熵损失函数
def CrossEntropy(y_hat, y, dim=1):
    if dim==0:
        return -np.sum(y*np.log(y_hat + 1e-8))/len(y)
    return -np.sum(y*np.log(y_hat + 1e-8))/len(y[0])  #防止溢出


#均方误差
def MSE(y_hat, y,dim=1):
    '''1对列计算, 0对行计算'''
    if dim==0:
        return MSE(y_hat.T, y.T)
    return np.sum((y-y_hat)**2)/len(y[0])

#绝对误差
def MAE(y_hat, y, dim=1):
    if dim==0:
        return MAE(y_hat.T,y.T)
    return np.sum(abs(y-y_hat))/len(y[0])

if __name__ == '__main__':
    a = np.array([[1,2],[7,4]])
    b = np.array([[2,1],[2,2]])
    print(CrossEntropy(b,a))
    print(-0.5*(np.log(2)+7*np.log(2)+4*np.log(2)))