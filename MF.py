'''
共现矩阵：
    R = { user1:[item1, item2,...],... }

隐向量：
    user_vec = {user: vec}
    item_vec = {item: vec}   vec.shape: [k,] k是超参数

object:
    min Σ(R[user_i][item_j]-user_vec_i·item_vec_j)^2 + gama*(user_vec_i^2 + item_vec_j^2)= Σe^2 +gama(...)
    其中Σ对所有的i和j求和, i是用户下标, j是该用户点击过的文章

梯度：
    dL/d(user_vec_i) = -2Σ(item_vec_j)*e + gama*user_vec_i
    dL/d(item_vec_j) = -2Σ(user_vec_i)*e + gama*item_vec_j
    e = Σ(R[user_i][item_j]-user_vec_i·item_vec_j)
'''
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm


#矩阵分解类
class MF(object):
    '''
        hidden: 隐向量维度
        alpha: 步长
        gama: 正则化系数, 采用l2正则
        max_iter: 最大迭代次数
    '''
    def __init__(self, hidden, alpha, gama, max_iter=100):
        self.hidden = hidden
        self.alpha = alpha
        self.gama = gama
        self.max_iter = max_iter
        self.user_vec = {}
        self.item_vec = {}
    def fit(self, R):
        '''
            R: { user1:{item1:score11, ...},... }
            R[user_i][item_j] 表示用户i对物品j的评分
        '''
        if not self.user_vec and not self.item_vec:
            #首次训练进行初始化, 生成hidden维的隐向量
            print('initialize...')
            for user in tqdm(R):
                self.user_vec[user] = np.random.randn(self.hidden)/np.sqrt(self.hidden)
                for item in R[user]:
                    self.item_vec.setdefault(item, np.random.randn(self.hidden)/np.sqrt(self.hidden))
        #随机梯度下降进行你和
        for step in tqdm(range(self.max_iter)):
            for user in R:
                for item in R[user]:
                    e = R[user][item] - np.dot(self.user_vec[user], self.item_vec[item])
                    self.user_vec[user] += self.alpha*(e*self.item_vec[item] - self.gama*self.user_vec[user])
                    self.item_vec[item] += self.alpha*(e*self.user_vec[user] - self.gama*self.item_vec[item])
            self.alpha *= 0.95
    def predict(self, user, item):
        return np.dot(self.user_vec[user], self.item_vec[item])


print('----------test----------')
a=    {
        'user_id':[1,2,2,3,3,4,4,5,5,6,7,7,9,9],
        'click_article_id':[4,2,4,3,4,1,4,1,4,4,1,4,1,4]
    } #简单的共现矩阵

#构造用户、item字典
def get_user_item_dict(df):
    '''输入用户行为df, 返回user_dict, item_dict, article_cnt
        return:
        user_dict: { user1:[item1,...],... }
        item_dict: { item1:[user1,...],... }
        article_cnt: list: [(article1, cnt1),....]
    '''
    user_dict, item_dict, article_cnt = {},{},{}
    #按行遍历, 将内容加入各dict中
    all_index = df.shape[0] #行数
    for index in tqdm(range(all_index)):
        data = df.loc[index]
        user, item = data['user_id'], data['click_article_id']
        #更新cnt
        if item not in article_cnt:
            article_cnt[item] = 1
        else: 
            article_cnt[item] += 1
        #更新user_dict
        if user not in user_dict:
            user_dict[user] = [item]
        else:
            user_dict[user].append(item)
        #更新item_dict
        if item not in item_dict:
            item_dict[item] = [user]
        else:
            item_dict[item].append(user)
    article_cnt = sorted(article_cnt.items(), key=lambda x:x[1], reverse=True) #按点击次数从大到小排序
    return user_dict, item_dict, article_cnt

#需要将用户字典转化为MF输入的格式
# {user1:[item1,...],...} → {user1:{item1:score1,...},...}
def to_input(dic):
    res = {}
    for user in dic:
        res[user] = {item:1 for item in dic[user]}
    return res

#模型训练
user_dict, item_dict, article_cnt = get_user_item_dict(pd.DataFrame(a))
R = to_input(user_dict) #MF的输入
mf = MF(10, 0.1, 0.01, 100)
mf.fit(R)
pre = {}
for i in range(1,6):
    pre[i] = {j:mf.predict(i,j) for j in item_dict}


print('原始打分：',R)
print('预测打分：',pre)
print('----------test----------')