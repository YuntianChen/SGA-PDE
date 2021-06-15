import numpy as np
import torch
import torch.nn as nn
import scipy.io as scio
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import random
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import configure as config

max_epoch = config.max_epoch
train_ratio = config.train_ratio # the ratio of training dataset
fine_ratio = config.fine_ratio # 通过MetaData加密数据的倍数
device = config.device
num_feature = config.num_feature

seed = config.seed
torch.random.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

# load data
u=config.u
x=config.x
t=config.t
n, m = u.shape
Y_raw = pd.DataFrame(u.reshape(-1,1))
X1 = np.repeat(x.reshape(-1,1), m, axis=1)
X2 = np.repeat(t.reshape(1,-1), n, axis=0)
X_raw_norm = pd.concat([pd.DataFrame(X1.reshape(-1,1)), pd.DataFrame(X2.reshape(-1,1))], axis=1, sort=False)
X = ((X_raw_norm-X_raw_norm.mean()) / (X_raw_norm.std()))
Y = ((Y_raw-Y_raw.mean()) / (Y_raw.std()))

# MetaData Train the NN model
def model_NN(x_train, y_train, num_feature):
    print('Using NN model')
    display_step = int(max_epoch/5)
    hidden_dim = 50
    x = torch.from_numpy(x_train.values).float()
    y = torch.from_numpy(y_train.values).float()
    x, y =Variable(x).to(device), Variable(y).to(device)
    # 训练模型
    model = config.Net(num_feature, hidden_dim, 1).to(device)
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum =0.9, weight_decay=1e-2)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), 5) # clip gradient
        optimizer.step()
        if (epoch+1)%display_step == 0:
            print('hi')
            # print('step %d, loss= %.6f'%(epoch+1, loss.cpu().data[0]))
    y_pred_train = model(x)
    y_pred_train = y_pred_train.cpu().data.numpy().flatten()
    return y_pred_train, model
# 划分数据集  
data_num = Y.shape[0]
total_ID = list(range(0,data_num))
def split(full_list,shuffle=False,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2
train_index, test_index = split(total_ID, shuffle=True, ratio = train_ratio)
x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]
print('index shape:')
print(np.shape(train_index))
print(np.shape(test_index))
# 建模
y_pred_train, model = model_NN(x_train, y_train, num_feature)
torch.save(model.state_dict(), config.path) 

# 进行预测
x_test, y_test = Variable(torch.from_numpy(x_test.values).float()).to(device), Variable(torch.from_numpy(y_test.values).float().to(device))
y_pred = model(x_test)
x_test, y_test = x_test.cpu().data.numpy().flatten(), y_test.cpu().data.numpy().flatten()
y_pred, y_pred_train = y_pred.cpu().data.numpy().flatten(), y_pred_train
# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
print('The rmse of training is:', mean_squared_error(y_train, y_pred_train) ** 0.5)

y_pred = y_pred.reshape(-1,1)
print('y_pred:',np.shape(y_pred))
# eval
def R2(y_test, y_pred):
    SSE = np.sum((y_test - y_pred)**2)          # 残差平方和
    SST = np.sum((y_test-np.mean(y_test))**2) #总体平方和
    R_square = 1 - SSE/SST # 相关性系数R^2
    return R_square
def eval_result (y_test, y_pred, y_train, y_pred_train):
    # eval
    print('Evaluating model performance...')
    y_test = np.array(y_test).reshape((-1,1)) #注意维度变换，才能让test和pred一致。y_test is [9] and y_pred is [9,1]，如果直接运算SSE则会为[9,9]再求和，就会错误。
    y_pred = np.array(y_pred).reshape((-1,1))
    y_train = np.array(y_train).reshape((-1,1)) #注意维度变换，才能让test和pred一致。y_test is [9] and y_pred is [9,1]，如果直接运算SSE则会为[9,9]再求和，就会错误。
    y_pred_train = np.array(y_pred_train).reshape((-1,1))
    print('The std(y_pred_train) is:',np.std(y_pred_train))
    if len(test_index) == 0:
        RMSE = 0
    else:
        RMSE = mean_squared_error(y_test, y_pred) ** 0.5
    RMSE_train = mean_squared_error(y_train, y_pred_train) ** 0.5
    print('The RMSE of prediction is:', RMSE)
    print('The RMSE of prediction of training dataset is:', RMSE_train)
    if len(test_index) == 0:
        R_square = 0
    else:
        R_square = R2(y_test, y_pred)
    R_square_train = R2(y_train, y_pred_train)
    print('The R2 is:', R_square)
    print('The R2 of training dataset is:', R_square_train)
    return RMSE, RMSE_train, R_square, R_square_train
RMSE, RMSE_train, R_square, R_square_train = eval_result (y_test, y_pred, y_train, y_pred_train)
result_test_real = y_test*Y_raw.std()[0]+Y_raw.mean()[0]
result_pred_real = y_pred*Y_raw.std()[0]+Y_raw.mean()[0]

print('Neural network generated')