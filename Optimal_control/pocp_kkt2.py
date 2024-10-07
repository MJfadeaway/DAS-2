# -*- coding: utf-8 -*-
"""
Created on Thu May 26 19:19:03 2022

@author: Pengfei
"""
## pinn+projection 不用u的显式表达式

import argparse
import torch
import numpy as np
import time
import data_pocp as data
import itertools
import matplotlib.pyplot as plt
import bfgs
from matplotlib import cm

np.random.seed(1234)
torch.manual_seed(1234)

#%% ----------------------------------------------------------------------------------------------------
# Neural network
class Net(torch.nn.Module):
    def __init__(self, layers, device):
        super(Net, self).__init__()
        self.layers = layers
        self.layers_hid_num = len(layers)-2
        self.device = device

        fc = []
        for i in range(self.layers_hid_num):
            fc.append(torch.nn.Linear(self.layers[i],self.layers[i+1]))
            fc.append(torch.nn.Linear(self.layers[i+1],self.layers[i+1]))
        fc.append(torch.nn.Linear(self.layers[-2],self.layers[-1]))
        self.fc = torch.nn.Sequential(*fc)
    
    def forward(self, x):
        for i in range(self.layers_hid_num):
            h = torch.sin(self.fc[2*i](x))
            h = torch.sin(self.fc[2*i+1](h))
            temp = torch.zeros(x.shape[0],self.layers[i+1]-self.layers[i]).to(self.device)
            x = h + torch.cat((x,temp),1)
        return self.fc[-1](x)

class LEN():
    def __init__(self):
        pass
    def forward(self,X):
        x = X[:,0]
        y = X[:,1]
        mu1 = X[:,2]
        L = x*(2-x)*y*(1-y)*((x-1.5)**2+(y-0.5)**2-mu1**2)
        return L.reshape(-1,1)
    
def data_plot(inner):
    # plot samples
    fig, ax = plt.subplots()
    ax.set_xlim(-0.1,4.1)
    ax.set_ylim(-0.1,1.1)
    ax.set_aspect('equal','box')
    inner = inner.cpu().detach().numpy()
    ax.scatter(inner[:,0],inner[:,1],s=1)
    #fig.tight_layout()
    plt.show()

#%%
# y,u,p 的表示
def pred_y(NetY, lenth, X):
    return NetY(X)*lenth.forward(X) + 1

def pred_u(NetU, X):
    return NetU(X)

def pred_p(NetP, lenth, X):
    return NetP(X)*lenth.forward(X)

# Loss function
def loss_func_kkt(NetY, NetP, NetU, lenth, InSet, BdSet):
    InSet.y = pred_y(NetY, lenth, InSet.x)
    InSet.p = pred_p(NetP, lenth, InSet.x)
    InSet.u = pred_u(NetU, InSet.x)
    
    yx, = torch.autograd.grad(InSet.y, InSet.x,
                              create_graph=True, retain_graph=True,
                              grad_outputs=InSet.weight_grad)
    yx0 = yx[:,0:1]
    yx1 = yx[:,1:2]
    yx0x, = torch.autograd.grad(yx0, InSet.x,
                                create_graph=True, retain_graph=True,
                                grad_outputs=InSet.weight_grad)
    yx0x0 = yx0x[:,0:1]
    yx1x, = torch.autograd.grad(yx1, InSet.x,
                                create_graph=True, retain_graph=True,
                                grad_outputs=InSet.weight_grad)
    yx1x1 = yx1x[:,1:2]

    res_y = -(yx0x0+yx1x1) - InSet.u

    px, = torch.autograd.grad(InSet.p, InSet.x,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=InSet.weight_grad)
    px0 = px[:,0:1]
    px1 = px[:,1:2]
    px0x, = torch.autograd.grad(px0, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    px0x0 = px0x[:,0:1]
    px1x, = torch.autograd.grad(px1, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    px1x1 = px1x[:,1:2]
    
    res_p = -(px0x0+px1x1) - (InSet.y - InSet.y_d)
        
              
    InSet.loss_pde_y = (res_y**2).mean() 
    InSet.loss_pde_p = (res_p**2).mean()
    InSet.grad = data.alpha*InSet.u + InSet.p

    InSet.loss_var = ((InSet.u - data.P(InSet.u - c*InSet.grad))**2).mean()
                 
    loss = InSet.loss_pde_y + InSet.loss_pde_p + InSet.loss_var   
    return loss**0.5



# optimality-conditions-loss
def KKT_loss(NetY, NetP, NetU, lenth, InSet, BdSet, loss_history, record=True):    
    # c is the step_size
    loss = loss_func_kkt(NetY, NetP, NetU, lenth, InSet, BdSet)

    if record:
        loss_history.append([((InSet.loss_pde_y)**0.5).item(), \
                             ((InSet.loss_pde_p)**0.5).item(), \
                               loss.item()])
            
    print('pde_y_loss: %.3e, pde_p_loss: %.3e, KKT_loss: %.3e' \
          %((InSet.loss_pde_y)**0.5, (InSet.loss_pde_p)**0.5, loss))
    return loss_history

# ----------------------------------------------------------------------------------------------------
# Train neural network
def train_kkt(NetY, NetP, NetU, lenth, InSet, BdSet,\
             Optim, epochs, epochs_i, loss_history): 
    print('Train Neural Network')
    
    # Record the optimal parameters
    loss = loss_func_kkt(NetY, NetP, NetU, lenth, InSet, BdSet).data
    print('epoch: %d, KKT_loss: %.3e, time: %.2f'
          %(0, loss, 0.00))

    # Training cycle
    for it in range(int(epochs/epochs_i)):
        start_time = time.time()

        # Forward and backward propogation
        def closure():
            Optim.zero_grad()
            loss = loss_func_kkt(NetY, NetP, NetU, lenth, InSet, BdSet)
            loss.backward()
            return loss
        Optim.step(closure)
        elapsed = time.time() - start_time
        print('epoch: %d, time: %.2f'%((it+1)*epochs_i, elapsed))        
        loss_history = KKT_loss(NetY, NetP, NetU, lenth, InSet, BdSet, loss_history)  
        
    return loss_history

def state_loss(x, NetY, NetU, lenth):
    x.requires_grad = True
    y = pred_y(NetY, lenth, x)
    u = pred_u(NetU, x)
    weight_grad = torch.ones(x.shape[0],1).to(x.device)
    yx, = torch.autograd.grad(y, x,
                              create_graph=True, retain_graph=True,
                              grad_outputs=weight_grad)
    yx0 = yx[:,0:1]
    yx1 = yx[:,1:2]
    yx0x, = torch.autograd.grad(yx0, x,
                                create_graph=True, retain_graph=True,
                                grad_outputs=weight_grad)
    yx0x0 = yx0x[:,0:1]
    yx1x, = torch.autograd.grad(yx1, x,
                                create_graph=True, retain_graph=True,
                                grad_outputs=weight_grad)
    yx1x1 = yx1x[:,1:2]

    res_y = -(yx0x0+yx1x1) - u  
    return (res_y**2).mean() 


#%% Configurations
parser = argparse.ArgumentParser(description='Least-Squares Neural Network Method')
parser.add_argument('--nx_in', type=int, default=[40000],
                    help='size the interior set')
parser.add_argument('--nx_te_in', type=int, default=[1000],
                    help='size of the test set')
parser.add_argument('--layers_y', type=int, default=[4,25,25,25,1],
                    help='network structure')
parser.add_argument('--layers_p', type=int, default=[4,25,25,25,1],
                    help='network structure')
parser.add_argument('--layers_u', type=int, default=[4,25,25,25,1],
                    help='network structure')
parser.add_argument('--epochs', type=int, default=50000,
                    help='number of iterations')
parser.add_argument('--epochs_i', type=int, default=100,
                    help='number of inner iterations')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--use_cuda', type=bool, default=True,
                    help='device')
parser.add_argument('--dtype', type=str, default='float64',
                    help='data type')
args = parser.parse_args()

use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if args.dtype=='float16':
    torch.set_default_dtype(torch.float16)
if args.dtype=='float32':
    torch.set_default_dtype(torch.float32)
if args.dtype=='float64':
    torch.set_default_dtype(torch.float64)


#%%
sampling = 'quasi-random'
nx_in = torch.tensor(args.nx_in).int().to(device)
nx_te_in = torch.tensor(args.nx_te_in).int().to(device)

# Parepare data set
in_set = data.InSet(nx_in, device, sampling)
bd_set = None
te_in_set = data.Te_InSet(nx_te_in, device)
te_bd_set = None
    
# construct length func
lenth = LEN()

c = 1/data.alpha
# c = c/10

# Construct neural network
net_y = Net(args.layers_y, device).to(device)
net_p = Net(args.layers_p, device).to(device)
net_u = Net(args.layers_u, device).to(device)
loss_history = []
loss_history = \
    KKT_loss(net_y, net_p, net_u, lenth, in_set, bd_set, loss_history)


# net_y = torch.load("./net_save/test3-pinn+proj-%s-ynet.pt"%c)
# net_u = torch.load('./net_save/test3-pinn+proj-%s-unet.pt'%c)
# net_p = torch.load('./net_save/test3-pinn+proj-%s-pnet.pt'%c)
# in_set.x = torch.load('./net_save/test3-pinn+proj-%s-x.pt'%c)
# loss_history = torch.load('./net_save/test3-pinn+proj-%s-loss.pt'%c).tolist()


optim = bfgs.BFGS(itertools.chain(net_y.parameters(),
                                  net_p.parameters(),
                                  net_u.parameters()),
                  lr=1, max_iter=args.epochs_i,
                  tolerance_grad=1e-16, tolerance_change=1e-16,
                  line_search_fn='strong_wolfe') 

#%%  Train neural network
start_time = time.time()
loss_history = train_kkt(net_y, net_p, net_u, lenth, in_set, bd_set,
          optim, args.epochs, args.epochs_i, loss_history)

  
elapsed = time.time() - start_time

print('Finishied! train time: %.2f\n' %(elapsed))


  #%%
# save nn module
loss_history = np.array(loss_history)
fname1 = "./net_save/test3-pinn+proj-%s-ynet.pt"%c
fname2 = "./net_save/test3-pinn+proj-%s-unet.pt"%c
fname3 = "./net_save/test3-pinn+proj-%s-pnet.pt"%c
fname4 = "./net_save/test3-pinn+proj-%s-x.pt"%c
fname5 = "./net_save/test3-pinn+proj-%s-loss.pt"%c

torch.save(net_y, fname1)
torch.save(net_u, fname2)
torch.save(net_p, fname3)
torch.save(in_set.x, fname4)
torch.save(loss_history, fname5)



"""

#%% Plot KKT-loss and obj_value history

fig, ax = plt.subplots(1,1,figsize=(6,4))

ax.semilogy(np.array(loss_history))
ax.legend(['state loss', 'adjoint loss', 'kkt loss'])
ax.set_xlabel('iterations') 

fig.tight_layout()
plt.grid()


#%% ------------------------------------------------------------------------------------------------
# Plot solution contour plot for fixed mu1 and mu2
n=100
mu1 = 0.25
mu2 = 2.0
num_line = 400

fig, ax = plt.subplots(2,1,figsize=(8,7))
for j in range(2):
    ax[j].axis('equal')
    ax[j].set_xlim([0,2])
    ax[j].set_ylim([0,1])
    ax[j].axis('off')

x0 = np.linspace(0,2,2*n)
x1 = np.linspace(0,1,n)
x0, x1 = np.meshgrid(x0,x1)

ind = ((x0-1.5)**2 + (x1-0.5)**2 >= mu1**2)
x0 = x0[ind].reshape(-1,1)
x1 = x1[ind].reshape(-1,1)

te_data_tra = torch.from_numpy(np.hstack([x0,x1])).to(device)
te_data = torch.zeros([te_data_tra.shape[0],4]).to(device)
te_data[:,0:2] = te_data_tra
te_data[:,2:3] = mu1
te_data[:,3:4] = mu2

u = pred_u(net_u, te_data).cpu().detach().numpy().flatten()
y = pred_y(net_y, lenth, te_data).cpu().detach().numpy().flatten()
p = pred_p(net_p, lenth, te_data).cpu().detach().numpy().flatten()
yd = data.yd(te_data).cpu().detach().numpy().flatten()

x0 = x0.flatten()
x1 = x1.flatten()

ax0 = ax[0].tricontourf(x0, x1, u, num_line, cmap='coolwarm')

ax[0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
#ax[0].contour(ax0, linewidths=0.6, colors='black')
fig.colorbar(ax0,ax=ax[0],boundaries=np.linspace(0,10,5))
ax[0].set_title('PINN: u')
# levels=np.linspace(-1.5,1.5,50)

ax1 = ax[1].tricontourf(x0, x1, p, num_line, alpha=1, cmap='coolwarm')
ax[1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
#ax[1].contour(ax1, linewidths=0.6, colors='black')
fig.colorbar(ax1,ax=ax[1])
ax[1].set_title('PINN: y')

plt.suptitle(r'$\mu_1=%.3f, \mu_2=%.3f$'%(mu1,mu2),fontsize=20)

fig.tight_layout()
plt.show()


#%%
mu1 = 0.3
mu2 = 2.5
ua=0.
ub=10.
area = 2 - np.pi*mu1**2

n = 100
x0 = np.linspace(0,2,2*n)
x1 = np.linspace(0,1,n)
x0, x1 = np.meshgrid(x0,x1)

ind = ((x0-1.5)**2 + (x1-0.5)**2 >= mu1**2)
x0 = x0[ind].reshape(-1,1)
x1 = x1[ind].reshape(-1,1)

te_data_tra = torch.from_numpy(np.hstack([x0,x1])).to(device)
te_data = torch.zeros([te_data_tra.shape[0],4]).to(device)
te_data[:,0:2] = te_data_tra
te_data[:,2:3] = mu1
te_data[:,3:4] = mu2

u = pred_u(net_u, te_data).cpu().detach().numpy()
y = pred_y(net_y, lenth, te_data).cpu().detach().numpy()
yd = data.yd(te_data).cpu().detach().numpy()

J = (0.5*(y-yd)**2 + 0.5*data.alpha*u**2 ).mean()*area
print('(mu1,mu2,ua,ub) = (%.2f,%.1f,%.2f,%.2f),J:%.5f'%(mu1,mu2,ua,ub,J))
#np.save('./net_save/y-pinn-proj-(%.2f,%.1f,%.2f,%.2f)-%s.npy'%(mu1,mu2,ua,ub,c),y)
np.save('./net_save/u-pinn-proj-(%.2f,%.1f,%.2f,%.2f)-%s.npy'%(mu1,mu2,ua,ub,c),u)



#%%
net_y = torch.load('./net_save/test3-pinn-ynet.pt')
net_u = torch.load('./net_save/test3-pinn-unet.pt')
mu1 = 0.25
mu2 = 2.5

n = 100
x0 = np.linspace(0,2,2*n)
x1 = np.linspace(0,1,n)
x0, x1 = np.meshgrid(x0,x1)

ind = ((x0-1.5)**2 + (x1-0.5)**2 >= mu1**2)
x0 = x0[ind].reshape(-1,1)
x1 = x1[ind].reshape(-1,1)

te_data_tra = torch.from_numpy(np.hstack([x0,x1])).to(device)
te_data = torch.zeros([te_data_tra.shape[0],4]).to(device)
te_data[:,0:2] = te_data_tra
te_data[:,2:3] = mu1
te_data[:,3:4] = mu2
ss = state_loss(te_data, net_y, net_u, lenth)
print(torch.sqrt(ss))


#%%
# Plot solution contour plot for diff mu1 and mu2(yup)
n=100
num_line = 400
mu1_list = [0.1,0.25,0.4]
mu2_list = [1.5,1.5,1.5]

fig, ax = plt.subplots(3,3,figsize=(11.5,7))
for i in range(3):
    for j in range(3):
        ax[i,j].axis('equal')
        ax[i,j].set_xlim([0,2])
        ax[i,j].set_ylim([0,1])
        ax[i,j].axis('off')

for j in range(3):
    mu1 = mu1_list[j]
    mu2 = mu2_list[j]
    x0 = np.linspace(0,2,2*n)
    x1 = np.linspace(0,1,n)
    x0, x1 = np.meshgrid(x0,x1)
    ind = ((x0-1.5)**2 + (x1-0.5)**2 >= mu1**2)
    x0 = x0[ind].reshape(-1,1)
    x1 = x1[ind].reshape(-1,1)
    te_data_tra = torch.from_numpy(np.hstack([x0,x1])).to(device)
    te_data = torch.zeros([te_data_tra.shape[0],4]).to(device)
    te_data[:,0:2] = te_data_tra
    te_data[:,2:3] = mu1
    te_data[:,3:4] = mu2
    u = pred_u(net_u, te_data).cpu().detach().numpy().flatten()
    y = pred_y(net_y, lenth, te_data).cpu().detach().numpy().flatten()
    p = pred_p(net_p, lenth, te_data).cpu().detach().numpy().flatten()
    x0 = x0.flatten()
    x1 = x1.flatten()
    
    ax0 = ax[0,j].tricontourf(x0, x1, u, num_line, cmap='coolwarm')
    ax[0,j].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
    fig.colorbar(ax0,ax=ax[0,j])
    
    ax1 = ax[1,j].tricontourf(x0, x1, y, num_line, alpha=1, cmap='coolwarm')
    ax[1,j].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
    fig.colorbar(ax1,ax=ax[1,j])
    
    ax2 = ax[2,j].tricontourf(x0, x1, p, num_line, alpha=1, cmap='coolwarm')
    ax[2,j].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
    fig.colorbar(ax2,ax=ax[2,j])

ax[0,0].set_title(r'$\mu_1=%.3f, \mu_2=%.3f$'%(mu1_list[0],mu2_list[0]),fontsize=15)
ax[0,1].set_title(r'$\mu_1=%.3f, \mu_2=%.3f$'%(mu1_list[1],mu2_list[1]),fontsize=15)
ax[0,2].set_title(r'$\mu_1=%.3f, \mu_2=%.3f$'%(mu1_list[2],mu2_list[2]),fontsize=15)

ax[0,0].text(-0.4,0.5,'u:',fontsize=15)
ax[1,0].text(-0.4,0.5,'y:',fontsize=15)
ax[2,0].text(-0.4,0.5,'p:',fontsize=15)

#fig.tight_layout()

#%%
# Plot solution contour plot for diff mu1 and mu2(u)

n=100
num_line = 400
mu1_list = [0.1,0.25,0.3]
mu2_list = [1.,2.,2.5]
from matplotlib.ticker import FuncFormatter
fmt = lambda x, pos: '{:.3f}'.format(x)

fig, ax = plt.subplots(3,3,figsize=(14,6))
for i in range(3):
    for j in range(3):
        ax[i,j].axis('equal')
        ax[i,j].set_xlim([0,2])
        ax[i,j].set_ylim([0,1])
        ax[i,j].axis('off')

for i in range(3):
    mu1 = mu1_list[i]
    x0 = np.linspace(0,2,2*n)
    x1 = np.linspace(0,1,n)
    x0, x1 = np.meshgrid(x0,x1)
    ind = ((x0-1.5)**2 + (x1-0.5)**2 >= mu1**2)
    x0 = x0[ind].reshape(-1,1)
    x1 = x1[ind].reshape(-1,1)
    te_data_tra = torch.from_numpy(np.hstack([x0,x1])).to(device)
    x0 = x0.flatten()
    x1 = x1.flatten()
    te_data = torch.zeros([te_data_tra.shape[0],4]).to(device)
    te_data[:,0:2] = te_data_tra
    te_data[:,2:3] = mu1
    for j in range(3):
        mu2 = mu2_list[j]
        te_data[:,3:4] = mu2
        u = pred_u(net_u, te_data).cpu().detach().numpy().flatten()
        #y = pred_y(net_y, lenth, te_data).cpu().detach().numpy().flatten()
        #p = pred_p(net_p, lenth, te_data).cpu().detach().numpy().flatten()  
        ax0 = ax[i,j].tricontourf(x0, x1, u, num_line, cmap='coolwarm')
        ax[i,j].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
        fig.colorbar(ax0,ax=ax[i,j],aspect=10,format=FuncFormatter(fmt))
        ax[i,j].set_title(r'$\bf{\mu}=(%.2f,%.2f)$'%(mu1,mu2))

        
"""