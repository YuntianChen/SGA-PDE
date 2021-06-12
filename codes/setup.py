import numpy as np
from PDE_find import Diff, Diff2, FiniteDiff
import Data_generator as Data_generator
import scipy.io as scio
from requests import get
from inspect import isfunction
import math
import pdb
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.nn import Linear,Tanh,Sequential
from torch.autograd import Variable
import configure as config
from configure import divide

simple_mode = True
see_tree = None
plot_the_figures = True
use_metadata = False
use_difference = True

if use_difference == True:
    use_autograd = False
    print('Using difference method')
else:
    use_autograd = True
    print('Using autograd method')

def cubic(inputs):
    return np.power(inputs, 3)

# def divide(up, down, eta=1e-10):
#     while np.any(down == 0):
#         down += eta
#     return up/down

def get_random_int(max_int):
    random_result = get('https://www.random.org/integers/?num=1&min=0&max={0}&col=1&base=10&format=plain&rnd=new'.format(max_int)).content
    try:
        int(random_result)
    except:
        print(random_result)
    return int(random_result)

# rand = get_random_int(1e6)
rand = config.seed #0
print('random seed:{}'.format(rand))
# 237204
np.random.seed(rand)
random.seed(rand)

# load Metadata
u = Data_generator.u
x = Data_generator.x
t = Data_generator.t
x_all = Data_generator.x_all
n, m = u.shape
dx = x[2]-x[1]
dt = t[1]-t[0]
# 扩充维度使得与u的size相同
x = np.tile(x, (m, 1)).transpose((1, 0))
x_all = np.tile(x_all, (m, 1)).transpose((1, 0))
t = np.tile(t, (n, 1))

# load Origin data
u_origin=config.u
x_origin=config.x
t_origin=config.t
n_origin, m_origin = u_origin.shape
dx_origin = x_origin[2]-x_origin[1]
dt_origin = t_origin[1]-t_origin[0]
# 扩充维度使得与u的size相同
x_origin = np.tile(x_origin, (m_origin, 1)).transpose((1, 0))
t_origin = np.tile(t_origin, (n_origin, 1))

# 差分
# calculate the error of correct cofs & correct terms
if use_difference == True:
    ut = np.zeros((n, m))
    for idx in range(n):
        ut[idx, :] = FiniteDiff(u[idx, :], dt)
    ux = np.zeros((n, m))
    uxx = np.zeros((n, m))
    uxxx = np.zeros((n, m))
    for idx in range(m):
        ux[:, idx] = FiniteDiff(u[:, idx], dx) #idx is the id of one time step
    for idx in range(m):
        uxx[:, idx] = FiniteDiff(ux[:, idx], dx)
    for idx in range(m):
        uxxx[:, idx] = FiniteDiff(uxx[:, idx], dx)

    ut_origin = np.zeros((n_origin, m_origin))
    for idx in range(n_origin):
        ut_origin[idx, :] = FiniteDiff(u_origin[idx, :], dt_origin)
    ux_origin = np.zeros((n_origin, m_origin))
    uxx_origin = np.zeros((n_origin, m_origin))
    uxxx_origin = np.zeros((n_origin, m_origin))
    for idx in range(m_origin):
        ux_origin[:, idx] = FiniteDiff(u_origin[:, idx], dx_origin) #idx is the id of one time step
    for idx in range(m_origin):
        uxx_origin[:, idx] = FiniteDiff(ux_origin[:, idx], dx_origin)
    for idx in range(m_origin):
        uxxx_origin[:, idx] = FiniteDiff(uxx_origin[:, idx], dx_origin)

# autograd 问题在于被求导的部分形式不确定，如果每次重新训练神经网，代价过高。
if use_autograd == True:
    # load model
    hidden_dim = config.hidden_dim
    num_feature = config.num_feature
    model = config.Net(num_feature, hidden_dim, 1)
    model.load_state_dict(torch.load(config.path))
    # autograd
    def fun(x,t,Net):
        database = torch.cat((x,t), 1)
        database = Variable(database, requires_grad=True)
        PINNstatic=Net(database.float())
        H_grad = torch.autograd.grad(outputs=PINNstatic.sum(), inputs=database, create_graph=True)[0]
        Ht = H_grad[:, 1]
        Hx = H_grad[:, 0]
        Ht_n = Ht.data.numpy()
        Hx_n=Hx.data.numpy()
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, 0]
        Hxx_n = Hxx.data.numpy()
        Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, 0]
        Hxxx_n = Hxxx.data.numpy()
        Htt = torch.autograd.grad(outputs=Ht.sum(), inputs=database, create_graph=True)[0][:, 1]
        Htt_n = Htt.data.numpy()
        return Hx_n, Hxx_n, Hxxx_n, Ht_n
    x_1d = np.reshape(x, (n*m, 1))
    t_1d = np.reshape(t, (n*m, 1))
    ux, uxx, uxxx, ut = fun(torch.from_numpy(x_1d), torch.from_numpy(t_1d), model)
    ut = np.reshape( ut, (n,m))
    ux = np.reshape( ux, (n,m))
    uxx = np.reshape( uxx, (n,m))
    uxxx = np.reshape( uxxx, (n,m))

    x_1d_origin = np.reshape(x_origin, (n_origin*m_origin, 1))
    t_1d_origin = np.reshape(t_origin, (n_origin*m_origin, 1))
    ux_origin, uxx_origin, uxxx_origin, ut_origin = fun(torch.from_numpy(x_1d_origin), torch.from_numpy(t_1d_origin), model)
    ut_origin = np.reshape( ut_origin, (n_origin,m_origin))
    ux_origin = np.reshape( ux_origin, (n_origin,m_origin))
    uxx_origin = np.reshape( uxx_origin, (n_origin,m_origin))
    uxxx_origin = np.reshape( uxxx_origin, (n_origin,m_origin))

# calculate error
exec (config.right_side)
exec (config.left_side)
n1, n2, m1, m2 = int(n*0.1), int(n*0.9), int(m*0), int(m*1)
right_side_full = right_side
right_side = right_side[n1:n2, m1:m2]
left_side = left_side[n1:n2, m1:m2]
right = np.reshape(right_side, ((n2-n1)*(m2-m1), 1))
left = np.reshape(left_side, ((n2-n1)*(m2-m1), 1))
diff = np.linalg.norm(left-right, 2)/((n2-n1)*(m2-m1))
print('data error without edges',diff)

exec (config.right_side_origin)
exec (config.left_side_origin)
n1_origin, n2_origin, m1_origin, m2_origin = int(n_origin*0.1), int(n_origin*0.9), int(m_origin*0), int(m_origin*1)
right_side_full_origin = right_side_origin
right_side_origin = right_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
left_side_origin = left_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
right_origin = np.reshape(right_side_origin, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
left_origin = np.reshape(left_side_origin, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
diff_origin = np.linalg.norm(left_origin-right_origin, 2)/((n2_origin-n1_origin)*(m2_origin-m1_origin))
print('data error_origin without edges',diff_origin)

exec (config.right_side)
exec (config.left_side)
n1, n2, m1, m2 = int(n*0), int(n*1), int(m*0), int(m*1)
right_side_full = right_side
right_side = right_side[n1:n2, m1:m2]
left_side = left_side[n1:n2, m1:m2]
right = np.reshape(right_side, ((n2-n1)*(m2-m1), 1))
left = np.reshape(left_side, ((n2-n1)*(m2-m1), 1))
diff = np.linalg.norm(left-right, 2)/((n2-n1)*(m2-m1))
print('data error',diff)

exec (config.right_side_origin)
exec (config.left_side_origin)
n1_origin, n2_origin, m1_origin, m2_origin = int(n_origin*0), int(n_origin*1), int(m_origin*0), int(m_origin*1)
right_side_full_origin = right_side_origin
right_side_origin = right_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
left_side_origin = left_side_origin[n1_origin:n2_origin, m1_origin:m2_origin]
right_origin = np.reshape(right_side_origin, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
left_origin = np.reshape(left_side_origin, ((n2_origin-n1_origin)*(m2_origin-m1_origin), 1))
diff_origin = np.linalg.norm(left_origin-right_origin, 2)/((n2_origin-n1_origin)*(m2_origin-m1_origin))
print('data error_origin',diff_origin)

# plot the figures
if plot_the_figures == True:
    from matplotlib.pyplot import MultipleLocator
    x1 = int(n_origin*0.1)
    x2 = int(n_origin*0.9)
    t1 = int(m_origin*0.1)
    t2 = int(m_origin*0.9)
    # Plot the flow field
    plt.figure(figsize=(10,3))
    mm1=plt.imshow(u, interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max(u_origin), vmin=np.min(u_origin))
    plt.colorbar().ax.tick_params(labelsize=16) 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Metadata Field', fontsize = 15)
    plt.savefig(config.problem + '_Metadata_field_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.figure(figsize=(10,3))
    mm1=plt.imshow(u_origin, interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max(u_origin), vmin=np.min(u_origin))
    plt.colorbar().ax.tick_params(labelsize=16) 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Original Field', fontsize = 15)
    plt.savefig(config.problem + '_Original_field_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    # Plot the PDE terms
    fig=plt.figure(figsize=(5,3))
    ax = fig.add_subplot(1, 1, 1)
    x_index = np.linspace(0,256, n_origin)
    x_index_fine = np.linspace(0,100, n)
    if use_metadata == True:
        plt.plot(x_index_fine, ut[:,int(m/2)], color='red', label = 'Metadata')
    plt.plot(x_index, ut_origin[:,int(m_origin/2)], color='blue', linestyle='--') #, label = 'Raw data'
    # plt.ylim(np.min(ut_origin[x1:x2,t1:t2]), np.max(ut_origin[x1:x2,t1:t2]))
    # plt.title('$U_t$ (Left side)')
    ax.set_ylabel('$U_t$', fontsize=18)
    ax.set_xlabel('x', fontsize=18)
    # plt.legend(loc='upper left')
    x_major_locator=MultipleLocator(32)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig(config.problem + '_Ut_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight') 

    fig=plt.figure(figsize=(5,3))
    ax = fig.add_subplot(1, 1, 1)
    x_index = np.linspace(0,256, n_origin)
    x_index_fine = np.linspace(0,100, n)
    if use_metadata == True:
        plt.plot(x_index_fine, ux[:,int(m/2)], color='red', label = 'Metadata')
    plt.plot(x_index, ux_origin[:,int(m_origin/2)], color='blue', linestyle='--') #, label = 'Raw data'
    # plt.ylim(np.min(ux_origin[x1:x2,t1:t2]), np.max(ux_origin[x1:x2,t1:t2]))
    # plt.title('$U_x$')
    ax.set_ylabel('$U_x$', fontsize=18)
    ax.set_xlabel('x', fontsize=18)
    # plt.legend(loc='upper left')
    x_major_locator=MultipleLocator(32)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig(config.problem + '_Ux_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight') 

    fig=plt.figure(figsize=(5,3))
    ax = fig.add_subplot(1, 1, 1)
    x_index = np.linspace(0,100, n_origin)
    x_index_fine = np.linspace(0,100, n)
    if use_metadata == True:
        plt.plot(x_index_fine, uxx[:,int(m/2)], color='red', label = 'Metadata')
    plt.plot(x_index, uxx_origin[:,int(m_origin/2)], color='blue', linestyle='--', label = 'Raw data')
    # plt.ylim(np.min(uxx_origin[x1:x2,t1:t2]), np.max(uxx_origin[x1:x2,t1:t2]))
    # plt.title('$U_x$'+'$_x$')
    ax.set_ylabel('$U_x$'+'$_x$', fontsize=18)
    ax.set_xlabel('x', fontsize=18)
    plt.legend(loc='upper left')
    plt.savefig(config.problem + '_Uxx_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight') 

    plt.figure(figsize=(5,3))
    x_index = np.linspace(0,100, n_origin)
    x_index_fine = np.linspace(0,100, n)
    if use_metadata == True:
        plt.plot(x_index_fine, u[:,int(m/2)], color='red', label = 'Metadata')
    plt.plot(x_index, u_origin[:,int(m_origin/2)], color='blue', linestyle='--', label = 'Raw data')
    # plt.ylim(np.min(u_origin[x1:x2,t1:t2]), np.max(u_origin[x1:x2,t1:t2]))
    plt.title('U')
    plt.legend(loc='upper left')
    plt.savefig(config.problem + '_U_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.figure(figsize=(5,3))
    x_index = np.linspace(0,100, (n2_origin-n1_origin))
    x_index_fine = np.linspace(0,100, (n2-n1))
    if use_metadata == True:
        plt.plot(x_index_fine, right_side[:,int((m2-m1)/2)], color='red', label = 'Metadata')
    plt.plot(x_index, right_side_origin[:,int((m2_origin-m1_origin)/2)], color='blue', linestyle='--', label = 'Raw data')
    # plt.ylim(np.min(right_side_origin[x1:x2,t1:t2]), np.max(right_side_origin[x1:x2,t1:t2]))
    plt.title('Right side')
    plt.legend(loc='upper left')
    plt.savefig(config.problem + '_Right_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.figure(figsize=(5,3))
    x_index = np.linspace(0,100, n_origin)
    x_index_fine = np.linspace(0,100, n)
    if use_metadata == True:
        plt.plot(x_index_fine, (ut-right_side_full)[:,int(m/2)], color='red', label = 'Metadata')
    plt.plot(x_index, (ut_origin-right_side_full_origin)[:,int(m_origin/2)], color='blue', linestyle='--', label = 'Raw data')
    # plt.ylim(np.min(ut_origin[x1:x2,t1:t2]), np.max(ut_origin[x1:x2,t1:t2]))
    plt.title('Residual')
    plt.legend(loc='upper left')
    plt.savefig(config.problem + '_Residual_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.figure(figsize=(10,3))
    mm1=plt.imshow((ut-right_side_full), interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max((ut_origin-right_side_full_origin)), vmin=np.min((ut_origin-right_side_full_origin)))
    # mm1=plt.imshow((ut-right_side), interpolation='nearest',  cmap='Blues', origin='lower', vmax=5, vmin=-5)
    plt.colorbar().ax.tick_params(labelsize=16) 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Metadata Residual', fontsize = 15)
    plt.savefig(config.problem + '_Metadata_Residual_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.figure(figsize=(10,3))
    mm1=plt.imshow((ut_origin-right_side_full_origin), interpolation='nearest',  cmap='Blues', origin='lower', vmax=np.max((ut_origin-right_side_full_origin)), vmin=np.min((ut_origin-right_side_full_origin)))
    # mm1=plt.imshow((ut_origin-right_side_origin), interpolation='nearest',  cmap='Blues', origin='lower', vmax=5, vmin=-5)
    plt.colorbar().ax.tick_params(labelsize=16) 
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Original Residual', fontsize = 15)
    plt.savefig(config.problem + '_Original_Residual_'+'%d'%(config.max_epoch/1000) + 'k.png', dpi = 300, bbox_inches='tight')

    plt.show()


###########################################################################################
# for default evaluation
default_u = np.reshape(u, (u.shape[0]*u.shape[1], 1))
default_ux = np.reshape(ux, (u.shape[0]*u.shape[1], 1))
default_uxx = np.reshape(uxx, (u.shape[0]*u.shape[1], 1))
# default_uxxx = np.reshape(uxxx, (u.shape[0]*u.shape[1], 1))
default_u2 = np.reshape(u**2, (u.shape[0]*u.shape[1], 1))
default_u3 = np.reshape(u**3, (u.shape[0]*u.shape[1], 1))
# default_terms = np.hstack((default_u, default_ux, default_uxx, default_u2, default_u3))
# default_names = ['u', 'ux', 'uxx', 'u^2', 'u^3']
# default_terms = np.hstack((default_u, default_ux))
# default_names = ['u', 'ux']
default_terms = np.hstack((default_u)).reshape(-1,1)
default_names = ['u']
print(default_terms.shape)
num_default = default_terms.shape[1]

zeros = np.zeros(u.shape)

if simple_mode:
    # ALL = np.array([['sin', 1, np.sin], ['cos', 1, np.cos], ['+', 2, np.add], ['-', 2, np.subtract],
    #                 ['*', 2, np.multiply], ['d', 2, Diff], ['u', 0, u], ['t', 0, t], ['x', 0, x]])
    # OPS = np.array([['sin', 1, np.sin], ['cos', 1, np.cos],
    #                 ['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], ['d', 2, Diff]])
    # OP1 = np.array([['sin', 1, np.sin], ['cos', 1, np.cos]])

    ALL = np.array([['+', 2, np.add], ['-', 2, np.subtract],['*', 2, np.multiply], ['/', 2, divide], ['d', 2, Diff], ['d^2', 2, Diff2], 
                    ['u', 0, u], ['x', 0, x], ['ux', 0, ux],  ['0', 0, zeros],
                    ['^2', 1, np.square], ['^3', 1, cubic]]) #  ['u^2', 0, u**2], ['uxx', 0, uxx], ['t', 0, t],
    OPS = np.array([['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], ['/', 2, divide],
                    ['d', 2, Diff], ['d^2', 2, Diff2], ['^2', 1, np.square], ['^3', 1, cubic]])
    ROOT = np.array([['*', 2, np.multiply], ['d', 2, Diff], ['d^2', 2, Diff2], ['/', 2, divide], ['^2', 1, np.square], ['^3', 1, cubic]])
    OP1 = np.array([['^2', 1, np.square], ['^3', 1, cubic]])
    OP2 = np.array([['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], ['/', 2, divide], ['d', 2, Diff], ['d^2', 2, Diff2]])
    # VARS = np.array([['u', 0, u], ['x', 0, x], ['0', 0, zeros], ['ux', 0, ux], ['uxx', 0, uxx], ['u^2', 0, u**2]]) 
    VARS = np.array([['u', 0, u], ['x', 0, x], ['0', 0, zeros], ['ux', 0, ux]])
    den = np.array([['x', 0, x]])

# else:
#     ALL = np.array([['sin', 1, np.sin], ['cos', 1, np.cos], ['log', 1, np.log], ['+', 2, np.add], ['-', 2, np.subtract],
#                     ['*', 2, np.multiply], ['/', 2, np.divide], ['d', 2, Diff], ['u', 0, u], ['t', 0, t], ['x', 0, x]])
#     OPS = np.array([['sin', 1, np.sin], ['cos', 1, np.cos], ['log', 1, np.log],
#                     ['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], ['/', 2, np.divide],
#                     ['d', 2, Diff]])
#     ROOT = np.array([['*', 2, np.multiply], ['d', 2, Diff]])
#     OP1 = np.array([['sin', 1, np.sin], ['cos', 1, np.cos], ['log', 1, np.log]])
#     OP2 = np.array(
#         [['+', 2, np.add], ['-', 2, np.subtract], ['*', 2, np.multiply], ['/', 2, np.divide], ['d', 2, Diff]])
#     VARS = np.array([['u', 0, u], ['t', 0, t], ['x', 0, x]])
#     den = np.array([['t', 0, t], ['x', 0, x]])

pde_lib, err_lib = [], []
