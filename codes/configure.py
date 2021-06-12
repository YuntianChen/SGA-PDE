import numpy as np
import torch
import scipy.io as scio
import torch
import torch.nn as nn

problem = 'chaffee-infante' # 'Burgers' # 'chaffee-infante' # 'Kdv' #'PDE_1'
seed = 0
device = torch.device('cuda:0')
# device = torch.device('cpu')

###########################################################################################
# Neural network
max_epoch = 100 * 1000
path = problem+'_sine_sin_50_3fc2_'+'%d'%(max_epoch/1000)+'k_Adam.pkl'
hidden_dim = 50

train_ratio = 1 # the ratio of training dataset
num_feature = 2
normal = True

###########################################################################################
# Metadata
fine_ratio = 2 # 通过MetaData加密数据的倍数
use_metadata = False
delete_edges = False
print('use_metadata =', use_metadata)
print('delete_edges =', delete_edges)

# AIC hyperparameter
aic_ratio = 1  # lower this ratio, less important is the number of elements to AIC value


print(path)
print('fine_ratio = ',fine_ratio)
###########################################################################################
class Net(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(int(n_hidden),n_output)
    def forward(self,x):
        out = torch.sin((self.fc1(x)))
        out = torch.sin((self.fc2(out)))
        out = torch.sin((self.fc2(out)))
        out = torch.sin((self.fc2(out)))
        out = self.predict(out) 
        return out

# Data
def divide(up, down, eta=1e-10):
    while np.any(down == 0):
        down += eta
    return up/down
# PDE-1: Ut= -Ux/x + 0.25Uxx
if problem == 'PDE_1':
    u=np.load("./data/PDE_1.npy").T
    nx = 100
    nt = 251
    x=np.linspace(1,2,nx)
    t=np.linspace(0,1,nt)
    right_side = 'right_side = -config.divide(ux, x) + 0.25*uxx'
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = -config.divide(ux_origin, x_all) + 0.25*uxx_origin'
    # right_side_origin = 'right_side_origin = -0.9979*config.divide(ux_origin, x_all) + 0.2498*uxx_origin'
    left_side_origin = 'left_side_origin = ut_origin'

# PDE-3: Ut= d(uux)(x)
if problem == 'PDE_3':
    u=np.load("./data/PDE_3.npy").T
    nx = 100
    nt = 251
    x=np.linspace(1,2,nx)
    t=np.linspace(0,0.5,nt)
    right_side = 'right_side = u*uxx + ux*ux'
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = u_origin*uxx_origin + ux_origin*ux_origin'
    # right_side_origin = 'right_side_origin = 0.9806*u_origin*uxx_origin + 0.9806*ux_origin*ux_origin'
    left_side_origin = 'left_side_origin = ut_origin'
    
    
# Burgers -u*ux+0.1*uxx
if problem == 'Burgers':
    data = scio.loadmat('./data/burgers_sine.mat')
    u=data.get("usol")
    x=np.squeeze(data.get("x"))
    t=np.squeeze(data.get("t").reshape(1,201))
    right_side = 'right_side = -u*ux+0.1*uxx'
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = -1*u_origin*ux_origin+0.1*uxx_origin'
    # right_side_origin = 'right_side_origin = -1.0011*u_origin*ux_origin+0.1024*uxx_origin'
    left_side_origin = 'left_side_origin = ut_origin'

# Burgers Rudy 虚数
# data = scio.loadmat('./data/burgers.mat')
# u = np.real(data['usol'])
# x = np.real(data['x'][0])
# t = np.real(data['t'])[:,0]
# right_side = 'right_side = -u*ux+0.1*uxx'
# left_side = 'left_side = ut'

# # Kdv -0.0025uxxx-uux
if problem == 'Kdv':
    data = scio.loadmat('./data/Kdv-PINN.mat')
    u=data.get("uu")
    x=np.squeeze(data.get("x"))
    t=np.squeeze(data.get("tt").reshape(1,201))
    right_side = 'right_side = -0.0025*uxxx-u*ux'
    # right_side = 'right_side = -0.5368*u + 0.0145*ux + -0.0091*uxx + 0.0064*(4*x*u + 5*x**2*ux + 8*x*ux + 4*u + 8*x**2*uxx + x**3*uxx + x**3*uxxx + 6*x*ux)'    #-0.5368u + 0.0145ux + -0.0091uxx + 0.0064((x * (((x * x) * u) d x)) d^2 x)
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = -0.0025*uxxx_origin-u_origin*ux_origin'
    # right_side_origin = 'right_side_origin = -0.0025*uxxx_origin-1.0004*u_origin*ux_origin'
    left_side_origin = 'left_side_origin = ut_origin'
# # Kdv -0.0025uxxx-uux  x方向高精度
# data = scio.loadmat('./data/KDV_SGA.mat')
# struct=data['KDV_PARA_RK4_CFD6']
# u=struct[0,0]['uu']
# x=np.squeeze(struct[0,0]['x'])
# t=np.squeeze(struct[0,0]['tt'])
# right_side = 'right_side = -0.0025*uxxx-u*ux'
# left_side = 'left_side = ut'

# KdV -0.0025*(1+0.25*sin(pi*x)*uxxx-uux
if problem == 'complex_Kdv':
    data = scio.loadmat('./data/kdv_para_0.25.mat')
    data=data.get('ukdv_para')
    u=data[0][0][1]
    x=data[0][0][2][0]
    t=data[0][0][0][0]
    right_side = 'right_side = -0.0025*(1+0.25*np.sin(np.pi*x))*uxxx-u*ux'
    left_side = 'left_side = ut'

# # parametric-convection-diffusion data:
if problem == 'convection-diffusion':
    un=np.load("parametric-convection-diffusion.npy")
    u = un.transpose((1, 0))
    x = np.array(torch.linspace(0,8,201))
    t = np.array(torch.linspace(0,4.98,250))
    right_side = 'right_side = -ux-0.25*np.sin(x)*ux+uxx'
    left_side = 'left_side = ut'

# chaffee-infante   u_t=u_xx-u+u**3
if problem == 'chaffee-infante': # 301*200的新数据
    u = np.load("./data/CI.npy")
    x = np.load("./data/x.npy")
    t = np.load("./data/t.npy") 
    # right_side = 'right_side = uxx-u+u**3'
    # right_side = 'right_side = 1.0002*uxx - 1.0008*u + 1.0004*u**3'
    right_side = 'right_side = - 1.0008*u + 1.0004*u**3'

    # right_side = 'right_side = -1.0855*u + 0.9985*uxx + 0.0906*u**2 + 0.9801*u**3'  #-1.0855u + 0.9985uxx + 0.0906u^2 + 0.9801u^3
    left_side = 'left_side = ut'
    right_side_origin = 'right_side_origin = uxx_origin-u_origin+u_origin**3'
    # right_side_origin = 'right_side_origin = 1.0002*uxx_origin-1.0008*u_origin+1.0004*u_origin**3'
    left_side_origin = 'left_side_origin = ut_origin'


