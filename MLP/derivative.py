#计算各种激活函数的偏导数
from functions import *

def derivative(F,x):
    '''返回F在x处的导数'''
    x.astype('float32')
    if F == sigmoid:
        return F(x)*(1-F(x))

    elif F == relu:
        if len(x.shape)==1:
            for i in range(len(x)):
                x[i] = 1 if x[i]>0 else 0
            return x
        else:
            for i in x:
                i = derivative(relu,i)
            return x
    
    elif F == tanh:
        if len(x.shape)==1:
            for i in range(len(x)):
                x[i] = 1 - x[i]**2
            return x
        else:
            for i in x:
                i = derivative(tanh,i)
            return x


if __name__ == '__main__':
    c = np.array([[-1,2],[-4,-3],[2,-1]])

    print(relu(c))

    print(derivative(relu,c))