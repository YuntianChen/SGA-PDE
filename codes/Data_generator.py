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

fine_ratio = config.fine_ratio # 通过MetaData加密数据的倍数
normal = config.normal
use_metadata = config.use_metadata

seed = config.seed
torch.random.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
device = config.device

u=config.u
x=config.x
t=config.t
n, m = u.shape
Y_raw = pd.DataFrame(u.reshape(-1,1))
X1 = np.repeat(x.reshape(-1,1), m, axis=1)
X2 = np.repeat(t.reshape(1,-1), n, axis=0)
X_raw_norm = pd.concat([pd.DataFrame(X1.reshape(-1,1)), pd.DataFrame(X2.reshape(-1,1))], axis=1, sort=False)


if use_metadata == True:
    # load model
    hidden_dim = config.hidden_dim
    num_feature = config.num_feature
    model = config.Net(num_feature, hidden_dim, 1).to(device)
    model.load_state_dict(torch.load(config.path, map_location=device))

    # generate new data
    n_fine = fine_ratio*n -1
    m_fine = fine_ratio*m -1
    x_new = np.linspace(x.min(), x.max(), n_fine)
    t_new = np.linspace(t.min(), t.max(), m_fine)
    X1 = np.repeat(x_new.reshape(-1,1), m_fine, axis=1)
    X2 = np.repeat(t_new.reshape(1,-1), n_fine, axis=0)
    X_raw_norm = pd.concat([pd.DataFrame(X1.reshape(-1,1)), pd.DataFrame(X2.reshape(-1,1))], axis=1, sort=False)
    if normal == True:
        X = ((X_raw_norm-X_raw_norm.mean()) / (X_raw_norm.std()))
    else:
        X = X_raw_norm
    y_pred = model(Variable(torch.from_numpy(X.values).float()).to(device))
    # y_pred = y_pred.cpu().data.numpy()
    y_pred = y_pred.cpu().data.numpy().flatten()
    if normal == True:
        result_pred_real = y_pred*Y_raw.std()[0]+Y_raw.mean()[0]
    else:
        result_pred_real = y_pred
    u_new = result_pred_real.reshape(n_fine, m_fine)

    u_new2 = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            u_new2[i,j] = u_new[i*2,j*2]

    diff =  (u - u_new2)/u_new2

    print(np.max(diff))
    print(np.min(diff))
    print(np.mean(np.abs(diff)))
    print(np.median(np.abs(diff)))

    plt.figure(figsize=(5,3))
    mm1=plt.imshow(np.abs(diff), interpolation='nearest',  cmap='Blues', origin='lower', vmax=0.05, vmin=0)
    plt.colorbar().ax.tick_params(labelsize=16) 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Data error', fontsize = 15)

    plt.figure(figsize=(5,3))
    plt.hist(diff.reshape(-1,1))

    plt.figure(figsize=(5,3))
    x_index = np.linspace(0,100, n)
    x_index_fine = np.linspace(0,100, n_fine)
    plt.plot(x_index, u[:,int(m/2)])
    plt.plot(x_index_fine, u_new[:,int(m_fine/2)])
    # plt.show()

if use_metadata == True:
    u = u_new
    x = x_new
    t = t_new
    
if use_metadata == False:
    u = u
    x = x
    t = t
    
x_all = x    
# 提取指定区间内的MetaData数据
if config.delete_edges == True:
    n, m = u.shape 
    u = u[int(n*0.1):int(n*0.9), int(m*0):int(m*1)]
    x = x[int(n*0.1):int(n*0.9)]
    t = t[int(m*0):int(m*1)]