import numpy as np
import matplotlib.pyplot as plt
def FiniteDiff(u, dx, d):
    # 用二阶微分计算d阶微分，不过在三阶以上准确性会比较低
    # u是需要被微分的数据
    # dx是网格的空间大小
    n = np.size(u)
    ux = np.zeros(n)
    if d == 1:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - u[i - 1]) / (2 * dx)

        ux[0] = (-3.0 / 2 * u[0] + 2 * u[1] - u[2] / 2) / dx
        ux[n - 1] = (2.0 / 2 * u[n - 1] - 2 * u[n - 2] + u[n - 3] / 2) / dx
        return ux
    if d == 2:
        for i in range(1, n - 1):
            ux[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx ** 2

        ux[0] = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx ** 2
        ux[n - 1] = (2 * u[n - 1] - 5 * u[n - 2] + 4 * u[n - 3] - u[n - 4]) / dx ** 2
        return ux
    if d == 3:
        for i in range(2, n - 2):
            ux[i] = (u[i + 2] / 2 - u[i + 1] + u[i - 1] - u[i - 2] / 2) / dx ** 3
        ux[0] = (-2.5 * u[0] + 9 * u[1] - 12 * u[2] + 7 * u[3] - 1.5 * u[4]) / dx ** 3
        ux[1] = (-2.5 * u[1] + 9 * u[2] - 12 * u[3] + 7 * u[4] - 1.5 * u[5]) / dx ** 3
        ux[n - 1] = (2.5 * u[n - 1] - 9 * u[n - 2] + 12 * u[n - 3] - 7 * u[n - 4] + 1.5 * u[n - 5]) / dx ** 3
        ux[n - 2] = (2.5 * u[n - 2] - 9 * u[n - 3] + 12 * u[n - 4] - 7 * u[n - 5] + 1.5 * u[n - 6]) / dx ** 3
        return ux
    if d > 3:
        return FiniteDiff(FiniteDiff(u, dx, 3), dx, d - 3)


un=np.load("PDE_3.npy")
print(un.shape)
nx = 100
nt = 251
x=np.linspace(1,2,nx)
t=np.linspace(0,0.5,nt)
dx=x[1]-x[0]
dt=t[1]-t[0]
u_x=np.zeros(un.shape)
u_xx=np.zeros(un.shape)
u_t=np.zeros(un.shape)
for i in range(un.shape[0]):
    u_x[i]=FiniteDiff(un[i],dx,1)
    u_xx[i] = FiniteDiff(un[i], dx, 2)
for i in range(un.shape[1]):
    u_t[:,i]=FiniteDiff(un.T[i],dt,1)
error=np.zeros(un.shape)
relavtive_error=np.zeros(un.shape)
for i in range(un.shape[0]):
    for j in range(un.shape[1]):
        #For PDE_1
        #error[i,j]=u_t[i,j]-0.25*u_xx[i,j]+u_x[i,j]*1/(x[j])
        #For PDE_2
        #error[i, j] = u_t[i, j] - 0.25 * u_xx[i, j] + u_x[i, j] * 1 / (x[j]+np.sin(np.pi*x[j]))
        #For PDE_3
        error[i, j] = u_t[i, j] - un[i,j]* u_xx[i, j] - u_x[i, j] **2
        relavtive_error[i,j]=error[i,j]/(u_t[i,j]+1e-8)
print(np.max(relavtive_error))
print(np.median(relavtive_error))

fig = plt.figure()
#定义画布为1*1个划分，并在第1个位置上进行作图
ax = fig.add_subplot(111)
im = ax.imshow(relavtive_error, cmap='Blues',vmin=-0.01,vmax=0.01)
#增加右侧的颜色刻度条
cbar=plt.colorbar(im)

fig = plt.figure()
#定义画布为1*1个划分，并在第1个位置上进行作图
ax = fig.add_subplot(111)
# im = ax.imshow(relavtive_error, cmap=plt.cm.rainbow,vmin=-0.01,vmax=0.01)
im = ax.imshow(un, cmap='Blues')
#增加右侧的颜色刻度条
cbar=plt.colorbar(im)

plt.show()