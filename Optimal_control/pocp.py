
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

# Error norm
def relative_error_l2(u, ua):
    return (((u-ua)**2).sum() / ((ua**2).sum()+1e-16)) ** 0.5

def relative_error_linf(u, ua):
    return abs(u-ua).max()/(abs(ua).max()+1e-16)

#%%
# y,u,p 的表示
def pred_y(NetY, lenth, X):
    return NetY(X)*lenth.forward(X) + 1

def pred_u(NetU, X):
    return NetU(X)

def pred_p(NetP, lenth, X):
    return NetP(X)*lenth.forward(X)

# Loss function
def loss_func_y(NetY, lenth, InSet, BdSet):
    InSet.y = pred_y(NetY, lenth, InSet.x)
    
    InSet.yx, = torch.autograd.grad(InSet.y, InSet.x,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=InSet.weight_grad)
    InSet.yx0 = InSet.yx[:,0:1]
    InSet.yx1 = InSet.yx[:,1:2]
    InSet.yx0x, = torch.autograd.grad(InSet.yx0, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.yx0x0 = InSet.yx0x[:,0:1]
    InSet.yx1x, = torch.autograd.grad(InSet.yx1, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.yx1x1 = InSet.yx1x[:,1:2]

    InSet.res_y = -(InSet.yx0x0+InSet.yx1x1) - InSet.u_vec
              
    InSet.loss_pde_y = (InSet.res_y**2).mean()
    return InSet.loss_pde_y**0.5

# Loss function
def loss_func_p(NetY, NetP, lenth, InSet, BdSet):
    InSet.y = pred_y(NetY, lenth, InSet.x)
    InSet.p = pred_p(NetP, lenth, InSet.x)
 
    InSet.px, = torch.autograd.grad(InSet.p, InSet.x,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=InSet.weight_grad)
    InSet.px0 = InSet.px[:,0:1]
    InSet.px1 = InSet.px[:,1:2]
    InSet.px0x, = torch.autograd.grad(InSet.px0, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.px0x0 = InSet.px0x[:,0:1]
    InSet.px1x, = torch.autograd.grad(InSet.px1, InSet.x,
                                      create_graph=True, retain_graph=True,
                                      grad_outputs=InSet.weight_grad)
    InSet.px1x1 = InSet.px1x[:,1:2]

    InSet.res_p = -(InSet.px0x0+InSet.px1x1) - (InSet.y - InSet.y_d)

    InSet.grad = data.alpha*InSet.u_vec + InSet.p

    InSet.loss_pde_p = (InSet.res_p**2).mean() 
           
    return InSet.loss_pde_p**0.5

def grad_step_u(InSet, c):
    InSet.u_vec = data.P(InSet.u_vec-c*InSet.grad).detach()

def loss_func_u(NetU, InSet):
    InSet.u = pred_u(NetU,  InSet.x)
    InSet.res_u = InSet.u-InSet.u_vec
    loss = (InSet.res_u**2).mean()
    return loss**0.5

# optimality-conditions-loss
def KKT_loss(NetY, NetP, U_vector, lenth, InSet, BdSet, c, loss_history, record=True):    
    # c is the step_size
    loss_y = loss_func_y(NetY, lenth, InSet, BdSet).item()
    loss_p = loss_func_p(NetY, NetP, lenth, InSet, BdSet).item()
    res_var = InSet.u_vec - data.P(InSet.u_vec-c*InSet.grad)
    loss_var = (((res_var**2).mean())**0.5).item()

    loss = (loss_var**2 + loss_y**2 + loss_p**2)**0.5
    if record:
        loss_history.append([loss_y, loss_p, loss_var, loss])
    print('state_loss: %.3e, adjoint_loss: %.3e, var_loss: %.3e, KKT_loss: %.3e' \
          %(loss_y, loss_p, loss_var, loss))
    return loss_history

# ----------------------------------------------------------------------------------------------------
# Train neural network
def train_y(NetY, lenth, InSet, BdSet,\
             Optim, epochs, epochs_i): 
    print('Train y Neural Network')
    
    # Record the optimal parameters
    loss = loss_func_y(NetY, lenth, InSet, BdSet).data
    print('epoch_y: %d, loss_y: %.3e, time: %.2f'
          %(0, loss, 0.00))
    # Training cycle
    for it in range(int(epochs/epochs_i)):
        start_time = time.time()

        # Forward and backward propogation
        def closure():
            Optim.zero_grad()
            loss = loss_func_y(NetY, lenth, InSet, BdSet)
            loss.backward()
            return loss
        Optim.step(closure)
        loss = loss_func_y(NetY, lenth, InSet, BdSet).data
        # Print
        elapsed = time.time() - start_time
        
        print('epoch_y: %d, loss_y: %.3e, time: %.2f'
              %((it+1)*epochs_i, loss, elapsed))

def train_p(NetY, NetP, lenth, InSet, BdSet,\
             Optim, epochs, epochs_i): 
    print('Train p Neural Network')
    
    # Record the optimal parameters
    loss = loss_func_p(NetY, NetP, lenth, InSet, BdSet).data
    print('epoch_p: %d, loss_p: %.3e, time: %.2f'
          %(0, loss, 0.00))


    # Training cycle
    for it in range(int(epochs/epochs_i)):
        start_time = time.time()

        # Forward and backward propogation
        def closure():
            Optim.zero_grad()
            loss = loss_func_p(NetY, NetP, lenth, InSet, BdSet)
            loss.backward()
            return loss
        Optim.step(closure)
        loss = loss_func_p(NetY, NetP, lenth, InSet, BdSet).data
        # Print
        elapsed = time.time() - start_time
        
        print('epoch_p: %d, loss_p: %.3e, time: %.2f'
              %((it+1)*epochs_i, loss, elapsed))

def train_u(NetU, InSet, Optim, epochs_u, epochs_i): 
    print('Train u Neural Network')

    loss = loss_func_u(NetU, InSet)
    print("epoch_u: {}, loss_u: {:.8f}".format(0,loss))
    
    for it in range(int(epochs_u/epochs_i)):   # epoch_i的倍数!!!
    
        def closure():
            loss = loss_func_u(NetU, InSet)
            Optim.zero_grad()
            loss.backward()
            return loss
        
        Optim.step(closure)    
        loss = loss_func_u(NetU, InSet)
        print("u_epoch: {}, loss: {:.8f}".format((it+1),loss))
     

#%% Configurations
# 10000, [4,20,20,20,1], 100, 200, 100 for first test
parser = argparse.ArgumentParser(description='Least-Squares Neural Network Method')
parser.add_argument('--nx_in', type=int, default=[40000],
                    help='size the interior set')
parser.add_argument('--nx_te_in', type=int, default=[2000],
                    help='size of the test set')
parser.add_argument('--layers_y', type=int, default=[4,25,25,25,1],
                    help='network structure')
parser.add_argument('--layers_p', type=int, default=[4,25,25,25,1],
                    help='network structure')
parser.add_argument('--layers_u', type=int, default=[4,25,25,25,1],
                    help='network structure')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of iterations')
parser.add_argument('--epochs_u', type=int, default=20000,
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

'''
# Construct neural network
net_y = Net(args.layers_y, device).to(device)
net_p = Net(args.layers_p, device).to(device)
net_u = Net(args.layers_u, device).to(device)
in_set.u_vec = pred_u(net_u,  in_set.x).detach()
c=100.
loss_history = []
loss_history = \
    KKT_loss(net_y, net_p, net_u, lenth, in_set, bd_set, c, loss_history)

'''
net_y = torch.load('./net_save/test3-pocp-ynet.pt')
net_u = torch.load('./net_save/test3-pocp-unet.pt')
net_p = torch.load('./net_save/test3-pocp-pnet.pt')
c=100*0.985**300
loss_history = torch.load('./net_save/test3-pocp-loss.pt').tolist()
in_set.u_vec =  torch.load('./net_save/test3-pocp-u_vec.pt')
in_set.x =  torch.load('./net_save/test3-pocp-x.pt')


optim_y = bfgs.BFGS(net_y.parameters(),
                  lr=1, max_iter=args.epochs_i,
                  tolerance_grad=1e-16, tolerance_change=1e-16,
                  line_search_fn='strong_wolfe')
optim_p = bfgs.BFGS(net_p.parameters(),
                  lr=1, max_iter=args.epochs_i,
                  tolerance_grad=1e-16, tolerance_change=1e-16,
                  line_search_fn='strong_wolfe')
optim_u = bfgs.BFGS(net_u.parameters(),
                  lr=1, max_iter=args.epochs_i,
                  tolerance_grad=1e-16, tolerance_change=1e-16,
                  line_search_fn='strong_wolfe')   


#%%  Train neural network
max_iters = 300
start_time = time.time()

for i in range(max_iters):
    print('\n    Iters: %d' %(i))

    train_y(net_y, lenth, in_set, bd_set,
              optim_y, args.epochs, args.epochs_i)
    train_p(net_y, net_p, lenth, in_set, bd_set,
              optim_p, args.epochs, args.epochs_i)
    optim_y = bfgs.BFGS(net_y.parameters(),
                      lr=1, max_iter=args.epochs_i,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')
    optim_p = bfgs.BFGS(net_p.parameters(),
                      lr=1, max_iter=args.epochs_i,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')
    grad_step_u(in_set, c)

    loss_history = \
        KKT_loss(net_y, net_p, net_u, lenth, in_set, bd_set, c, loss_history)
 
    c *= 0.985
    args.epochs += 2
    #args.epochs_u += 4

train_u(net_u, in_set, optim_u, args.epochs_u, args.epochs_i)   


elapsed = time.time() - start_time


#%%
# save nn module
loss_history = np.array(loss_history)

fname1 = "./net_save/test3-pocp-ynet.pt"
fname2 = "./net_save/test3-pocp-unet.pt"
fname3 = "./net_save/test3-pocp-pnet.pt"
fname4 = "./net_save/test3-pocp-loss.pt"
fname5 = "./net_save/test3-pocp-u_vec.pt"
fname6 = "./net_save/test3-pocp-x.pt"

torch.save(net_y, fname1)
torch.save(net_u, fname2)
torch.save(net_p, fname3)
torch.save(loss_history, fname4)
torch.save(in_set.u_vec, fname5)
torch.save(in_set.x, fname6)


#%%
net_y = torch.load('./net_save/test3-pocp-ynet.pt')
net_u = torch.load('./net_save/test3-pocp-unet.pt')
net_p = torch.load('./net_save/test3-pocp-pnet.pt')
loss_history = torch.load('./net_save/test3-pocp-loss.pt').tolist()
in_set.u_vec =  torch.load('./net_save/test3-pocp-u_vec.pt')
in_set.x =  torch.load('./net_save/test3-pocp-x.pt')


#%% Plot KKT-loss and obj_value history

fig, ax = plt.subplots(1,1,figsize=(6,4))

ax.semilogy(np.array(loss_history)[:,0:3])
ax.legend(['state loss', 'adjoint loss', 'variation loss'])
ax.set_xlabel('iterations') 

fig.tight_layout()
plt.grid()
#plt.savefig('./pic/pocp-loss.png', dpi=300)



#%%
# 3D plot training data

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(16,8))


xx = in_set.x.detach().cpu().numpy()
#ax.set_box_aspect((np.ptp(xx[:,0:1]), np.ptp(xx[:,1:2]), np.ptp(xx[:,2:3])))
ax.set_box_aspect((np.ptp(xx[:,0:1]), np.ptp(xx[:,1:2]), np.ptp(xx[:,1:2])))

s = ax.scatter(xx[:,0:1],xx[:,1:2],xx[:,2:3],s=0.5,c=xx[:,3:4],cmap='coolwarm')
#ax.scatter(-100,-100,100,s=10,label='training data')  # no use

ax.set_xlim3d(0,2)
ax.set_ylim3d(0,1)
ax.set_zlim3d(0.05,0.45)
ax.set_xlabel('x',fontsize=15)
ax.set_ylabel('y',fontsize=15)
ax.set_zlabel(r'$\mu_1$',fontsize=15)#,rotation=120)
ax.set_xticks([0,0.5,1,1.5,2])
ax.set_yticks([0,0.5,1])
#ax.legend()

#cbaxes = fig.add_axes([0.25, 0.85, 0.59, 0.03])
#cb = fig.colorbar(s,cax=cbaxes,orientation='horizontal',fraction=0.04)

cbaxes = fig.add_axes([0.75, 0.21, 0.016, 0.56])
cb = fig.colorbar(s, cax=cbaxes)
cb.set_label(r'$\mu_2$',fontsize=15)

cc='black'
x = np.linspace(0,1,50)
y = 0*x
z = 0*x+0.05
ax.plot(2*x,y,z,c=cc); ax.plot(2*x,y+1,z,c=cc); ax.plot(y,x,z,c=cc); ax.plot(y+2,x,z,c=cc)
ax.plot(2*x,y,z+0.4,c=cc); ax.plot(2*x,y+1,z+0.4,c=cc); ax.plot(y,x,z+0.4,c=cc); ax.plot(y+2,x,z+0.4,c=cc)
ax.plot(y,y,x*0.4+0.05,c=cc); ax.plot(y,y+1,x*0.4+0.05,c=cc)
ax.plot(y+2,y,x*0.4+0.05,c=cc); ax.plot(y+2,y+1,x*0.4+0.05,c=cc)

x1 = np.cos(2*np.pi*x); x2 = np.sin(2*np.pi*x)
k=20
for i in range(k):
    mu1 = 0.05+0.4*i/(k-1)
    ax.plot(x1*mu1+1.5,x2*mu1+0.5,y+mu1,c=cc)

#ax.set_title('Training data',fontsize=15)
#ax.view_init(40, 40)
#fig.tight_layout()
#position=fig.add_axes([0.15, 0.05, 0.7, 0.03])
#fraction=0.03
plt.savefig('./pic/pocp-data2.png', dpi=300)

#%% ------------------------------------------------------------------------------------------------
# Plot solution contour plot for fixed mu1 and mu2
n=100
mu1 = 0.25
mu2 = 2.0
num_line = 400

from matplotlib.ticker import FuncFormatter
fmt = lambda x, pos: '{:.3f}'.format(x)
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
fig.colorbar(ax0,ax=ax[0],format=FuncFormatter(fmt))
ax[0].set_title('AONN: u')
# levels=np.linspace(-1.5,1.5,50)

ax1 = ax[1].tricontourf(x0, x1, y, num_line, alpha=1, cmap='coolwarm')
ax[1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
#ax[1].contour(ax1, linewidths=0.6, colors='black')
fig.colorbar(ax1,ax=ax[1],format=FuncFormatter(fmt))
ax[1].set_title('AONN: y')

plt.suptitle(r'$\mu_1=%.3f, \mu_2=%.3f$'%(mu1,mu2),fontsize=20)

fig.tight_layout()
plt.show()
#cbar = plt.colorbar(format=FuncFormatter(fmt))



 #%% ------------------------------------------------------------------------------------------------
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
#plt.show()

#%%
# 3D Plot solution contour plot for fixed mu1 and mu2
fig = plt.figure(figsize=plt.figaspect(0.3))


ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.set_title('approximate: u')
ax.set_box_aspect((np.ptp(x0), np.ptp(x1), np.ptp(u)))

surf = ax.plot_surface(x0, x1, u, cmap=cm.coolwarm,
                       linewidth=1, antialiased=False)

ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.set_title('approximate: y')
ax.set_box_aspect((np.ptp(x0), np.ptp(x1), np.ptp(u)))
surf = ax.plot_surface(x0, x1, y, cmap=cm.coolwarm,
                       linewidth=1, antialiased=False)

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.set_title('desired state y_d')
ax.set_box_aspect((np.ptp(x0), np.ptp(x1), np.ptp(u)))
surf = ax.plot_surface(x0, x1, yd, cmap=cm.coolwarm,
                       linewidth=1, antialiased=False)



#%% ------------------------------------------------------------------------------------------------
# Plot solution contour plot for diff mu1 and mu2(u)
n=100
num_line = 400
mu1_list = [0.1,0.25,0.4]
mu2_list = [1.,1.5,2.]
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
#plt.savefig('./pic/test3-1.png', dpi=300)

#fig.tight_layout()

#%%
# difference between AONN PINN adn dolfin for fixed mu1 and mu2(with diff)
import matplotlib as mpl

y_dal = np.load('./net_save/y-dal-(0.25,2.0,0.00,10.00).npy').flatten()
u_dal = np.load('./net_save/u-dal-(0.25,2.0,0.00,10.00).npy').flatten()

y_pinn = np.load('./net_save/y-pinn2.1-(0.25,2.0,0.00,10.00).npy').flatten()
u_pinn = np.load('./net_save/u-pinn2.1-(0.25,2.0,0.00,10.00).npy').flatten()

#y_pinn = np.load('./net_save/y-pinn-(0.25,2.0,0.00,10.00).npy').flatten()
#u_pinn = np.load('./net_save/u-pinn-(0.25,2.0,0.00,10.00).npy').flatten()

#p_np = np.load('p-dal-(0.25,2.5).npy').flatten()
n=100
num_line = 400
mu1 = 0.25
mu2 = 2.0
fs = 15
from matplotlib.ticker import FuncFormatter
fmt = lambda x, pos: '{:.3f}'.format(x)
fig, ax = plt.subplots(3,4,figsize=(20,7))
for i in range(3):
    for j in range(4):
        ax[i,j].axis('equal')
        ax[i,j].set_xlim([0,2])
        ax[i,j].set_ylim([0,1])
        ax[i,j].axis('off')

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

norm1 = mpl.colors.Normalize(vmin=min(u_dal.min(),u.min()),\
                             vmax=max(u_dal.max(),u.max()))
norm2 = mpl.colors.Normalize(vmin=min(y_dal.min(),y.min()),\
                             vmax=max(y_dal.max(),y.max()))
    

ax00 = ax[0,0].tricontourf(x0, x1, u_dal/2, num_line, cmap='coolwarm')
ax_tmp = plt.tricontourf(x0, x1, u_dal, num_line, cmap='coolwarm', norm=norm1)
ax[0,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax_tmp,ax=ax[0,0],aspect=10,format=FuncFormatter(fmt))
ax[0,0].set_title('dolfin-adjoint: u',fontsize=fs)


ax02 = ax[0,2].tricontourf(x0, x1, y_dal, num_line, alpha=1, cmap='coolwarm', norm=norm2)
ax[0,2].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax02,ax=ax[0,2],aspect=10,format=FuncFormatter(fmt))
ax[0,2].set_title('dolfin-adjoint: y',fontsize=fs)

ax10 = ax[1,0].tricontourf(x0, x1, u, num_line, cmap='coolwarm', norm=norm1)
ax[1,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax10,ax=ax[1,0],aspect=10,format=FuncFormatter(fmt))
ax[1,0].set_title('AONN: u',fontsize=fs)

ax11 = ax[1,1].tricontourf(x0, x1, abs(u-u_dal), num_line, alpha=1, cmap='Oranges')
ax[1,1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax11,ax=ax[1,1],aspect=10,format=FuncFormatter(fmt))
ax[1,1].set_title('AONN difference: u',fontsize=fs)

ax20 = ax[2,0].tricontourf(x0, x1, u_pinn, num_line, cmap='coolwarm')#, norm=norm1)
ax[2,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax20,ax=ax[2,0],aspect=10)
ax[2,0].set_title('PINN+Projection: u',fontsize=fs)

ax21 = ax[2,1].tricontourf(x0, x1, abs(u_pinn-u_dal), num_line, alpha=1, cmap='Oranges')#, norm=norm2)
ax[2,1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax21,ax=ax[2,1],aspect=10)
ax[2,1].set_title('PINN+Projection difference: u',fontsize=fs)


ax12 = ax[1,2].tricontourf(x0, x1, y, num_line, alpha=1, cmap='coolwarm')#, norm=norm2)
ax[1,2].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax12,ax=ax[1,2],aspect=10)
ax[1,2].set_title('AONN: y',fontsize=fs)

ax13 = ax[1,3].tricontourf(x0, x1, abs(y-y_dal), num_line, alpha=1, cmap='Oranges')#, norm=norm2)
ax[1,3].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax13,ax=ax[1,3],aspect=10)
ax[1,3].set_title('AONN difference: y',fontsize=fs)

ax22 = ax[2,2].tricontourf(x0, x1, y_pinn, num_line, alpha=1, cmap='coolwarm')#, norm=norm2)
ax[2,2].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax22,ax=ax[2,2],aspect=10)
ax[2,2].set_title('PINN+Projection: y',fontsize=fs)

ax23 = ax[2,3].tricontourf(x0, x1, abs(y_pinn-y_dal), num_line, alpha=1, cmap='Oranges')#, norm=norm2)
ax[2,3].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax23,ax=ax[2,3],aspect=10)
ax[2,3].set_title('PINN+Projection difference: y',fontsize=fs)

#plt.suptitle(r'$\mu_1=%.2f, \mu_2=%.2f$'%(mu1,mu2),fontsize=20)
#fig.tight_layout()

#plt.savefig('./pic/test3-diff.png', dpi=300)

#%%
# difference between AONN PINN adn dolfin for fixed mu1 and mu2(with diff)
# another order picture
import matplotlib as mpl
c = 1/data.alpha
c = c*10
y_dal = np.load('./net_save/y-dal-(0.25,2.0,0.00,10.00).npy').flatten()
u_dal = np.load('./net_save/u-dal-(0.25,2.0,0.00,10.00).npy').flatten()

#y_pinnp = np.load('./net_save/y-pinn2.1-(0.25,2.0,0.00,10.00).npy').flatten()
#u_pinnp = np.load('./net_save/u-pinn2.1-(0.25,2.0,0.00,10.00).npy').flatten()

y_pinnp = np.load('./net_save/y-pinn-proj-(0.25,2.0,0.00,10.00)-%s.npy'%c).flatten()
u_pinnp = np.load('./net_save/u-pinn-proj-(0.25,2.0,0.00,10.00)-%s.npy'%c).flatten()

y_pinn = np.load('./net_save/y-pinn-kkt-(0.25,2.0,0.00,10.00).npy').flatten()
u_pinn = np.load('./net_save/u-pinn-kkt-(0.25,2.0,0.00,10.00).npy').flatten()


#y_pinn = np.load('./net_save/y-pinn-(0.25,2.0,0.00,10.00).npy').flatten()
#u_pinn = np.load('./net_save/u-pinn-(0.25,2.0,0.00,10.00).npy').flatten()

#p_np = np.load('p-dal-(0.25,2.5).npy').flatten()
n=100
num_line = 400
mu1 = 0.25
mu2 = 2.0
fs = 15
from matplotlib.ticker import FuncFormatter
fmt = lambda x, pos: '{:.3f}'.format(x)
fmt2 = lambda x, pos: '{:.1e}'.format(x)
fig, ax = plt.subplots(4,3,figsize=(16,9))
for i in range(4):
    for j in range(3):
        ax[i,j].axis('equal')
        ax[i,j].set_xlim([0,2])
        ax[i,j].set_ylim([0,1])
        ax[i,j].axis('off')
ax = ax.transpose()

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

norm1 = mpl.colors.Normalize(vmin=min(u_dal.min(),u.min()),\
                             vmax=max(u_dal.max(),u.max()))
norm2 = mpl.colors.Normalize(vmin=min(y_dal.min(),y.min()),\
                             vmax=max(y_dal.max(),y.max()))
    

ax00 = ax[0,0].tricontourf(x0, x1, u_dal/2, num_line, cmap='coolwarm')
ax_tmp = plt.tricontourf(x0, x1, u_dal, num_line, cmap='coolwarm', norm=norm1)
ax[0,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax_tmp,ax=ax[0,0],aspect=10,format=FuncFormatter(fmt))
ax[0,0].set_title('dolfin-adjoint: u',fontsize=fs)


ax02 = ax[0,2].tricontourf(x0, x1, y_dal, num_line, alpha=1, cmap='coolwarm', norm=norm2)
ax[0,2].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax02,ax=ax[0,2],aspect=10,format=FuncFormatter(fmt))
ax[0,2].set_title('dolfin-adjoint: y',fontsize=fs)


###
ax01 = ax[0,1].tricontourf(x0, x1, u_pinn, num_line, cmap='coolwarm', norm=norm1)
ax[0,1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax01,ax=ax[0,1],aspect=10,format=FuncFormatter(fmt))
ax[0,1].set_title('PINN: u',fontsize=fs)
###
ax03 = ax[0,3].tricontourf(x0, x1, y_pinn, num_line, cmap='coolwarm', norm=norm2)
ax[0,3].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax03,ax=ax[0,3],aspect=10,format=FuncFormatter(fmt))
ax[0,3].set_title('PINN: y',fontsize=fs)
###


ax10 = ax[1,0].tricontourf(x0, x1, u, num_line, cmap='coolwarm', norm=norm1)
ax[1,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax10,ax=ax[1,0],aspect=10,format=FuncFormatter(fmt))
ax[1,0].set_title('AONN: u',fontsize=fs)

ax11 = ax[1,1].tricontourf(x0, x1, abs(u-u_dal), num_line, alpha=1, cmap='Oranges')
ax[1,1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax11,ax=ax[1,1],aspect=10,format=FuncFormatter(fmt2))
ax[1,1].set_title('AONN error: u',fontsize=fs)

ax20 = ax[2,0].tricontourf(x0, x1, u_pinnp, num_line, cmap='coolwarm')#, norm=norm1)
ax[2,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax20,ax=ax[2,0],aspect=10,format=FuncFormatter(fmt))
ax[2,0].set_title('PINN+Projection: u',fontsize=fs)

ax21 = ax[2,1].tricontourf(x0, x1, abs(u_pinnp-u_dal), num_line, alpha=1, cmap='Oranges')#, norm=norm2)
ax[2,1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax21,ax=ax[2,1],aspect=10,format=FuncFormatter(fmt2))
ax[2,1].set_title('PINN+Projection error: u',fontsize=fs)


ax12 = ax[1,2].tricontourf(x0, x1, y, num_line, alpha=1, cmap='coolwarm')#, norm=norm2)
ax[1,2].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax12,ax=ax[1,2],aspect=10,format=FuncFormatter(fmt))
ax[1,2].set_title('AONN: y',fontsize=fs)

ax13 = ax[1,3].tricontourf(x0, x1, abs(y-y_dal), num_line, alpha=1, cmap='Oranges')#, norm=norm2)
ax[1,3].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax13,ax=ax[1,3],aspect=10,format=FuncFormatter(fmt2))
ax[1,3].set_title('AONN error: y',fontsize=fs)

ax22 = ax[2,2].tricontourf(x0, x1, y_pinnp, num_line, alpha=1, cmap='coolwarm')#, norm=norm2)
ax[2,2].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax22,ax=ax[2,2],aspect=10,format=FuncFormatter(fmt))
ax[2,2].set_title('PINN+Projection: y',fontsize=fs)

ax23 = ax[2,3].tricontourf(x0, x1, abs(y_pinnp-y_dal), num_line, alpha=1, cmap='Oranges')#, norm=norm2)
ax[2,3].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax23,ax=ax[2,3],aspect=10,format=FuncFormatter(fmt2))
ax[2,3].set_title('PINN+Projection error: y',fontsize=fs)

#plt.suptitle(r'$\mu_1=%.2f, \mu_2=%.2f$'%(mu1,mu2),fontsize=20)
#fig.tight_layout()

#plt.savefig('./pic/test3-diff-v2.png', dpi=300)
#%%
# difference between DAL and DAL-Net for fixed mu1 and mu2
import matplotlib as mpl

y_dal = np.load('./net_save/y-dal-(0.40,2.0,0.00,10.00).npy').flatten()
u_dal = np.load('./net_save/u-dal-(0.40,2.0,0.00,10.00).npy').flatten()

y_pinn = np.load('./net_save/y-pinn2.1-(0.40,2.0,0.00,10.00).npy').flatten()
u_pinn = np.load('./net_save/u-pinn2.1-(0.40,2.0,0.00,10.00).npy').flatten()

#y_pinn = np.load('./net_save/y-pinn-(0.25,2.0,0.00,10.00).npy').flatten()
#u_pinn = np.load('./net_save/u-pinn-(0.25,2.0,0.00,10.00).npy').flatten()

#p_np = np.load('p-dal-(0.25,2.5).npy').flatten()
n=100
num_line = 400
mu1 = 0.40
mu2 = 2.0
from matplotlib.ticker import FuncFormatter
fmt = lambda x, pos: '{:.3f}'.format(x)
fig, ax = plt.subplots(3,2,figsize=(11,7))
for i in range(3):
    for j in range(2):
        ax[i,j].axis('equal')
        ax[i,j].set_xlim([0,2])
        ax[i,j].set_ylim([0,1])
        ax[i,j].axis('off')

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

norm1 = mpl.colors.Normalize(vmin=min(u_dal.min(),u.min()),\
                             vmax=max(u_dal.max(),u.max()))
norm2 = mpl.colors.Normalize(vmin=min(y_dal.min(),y.min()),\
                             vmax=max(y_dal.max(),y.max()))
    
ax00 = ax[0,0].tricontourf(x0, x1, u_dal/2, num_line, cmap='coolwarm')
ax_tmp = plt.tricontourf(x0, x1, u_dal, num_line, cmap='coolwarm', norm=norm1)
ax[0,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax_tmp,ax=ax[0,0],aspect=10,format=FuncFormatter(fmt))
ax[0,0].set_title('dolfin-adjoint: u')

ax01 = ax[0,1].tricontourf(x0, x1, y_dal, num_line, alpha=1, cmap='coolwarm', norm=norm2)
ax[0,1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax01,ax=ax[0,1],aspect=10,format=FuncFormatter(fmt))
ax[0,1].set_title('dolfin-adjoint: y')

ax10 = ax[1,0].tricontourf(x0, x1, u, num_line, cmap='coolwarm', norm=norm1)
ax[1,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax10,ax=ax[1,0],aspect=10,format=FuncFormatter(fmt))
ax[1,0].set_title('AONN: u')

ax11 = ax[1,1].tricontourf(x0, x1, y, num_line, alpha=1, cmap='coolwarm', norm=norm2)
ax[1,1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax11,ax=ax[1,1],aspect=10,format=FuncFormatter(fmt))
ax[1,1].set_title('AONN: y')

ax20 = ax[2,0].tricontourf(x0, x1, u_pinn, num_line, cmap='coolwarm')#, norm=norm1)
ax[2,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax20,ax=ax[2,0],aspect=10)
ax[2,0].set_title('PINN: u')

ax21 = ax[2,1].tricontourf(x0, x1, y_pinn, num_line, alpha=1, cmap='coolwarm')#, norm=norm2)
ax[2,1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax21,ax=ax[2,1],aspect=10)
ax[2,1].set_title('PINN: y')

#plt.suptitle(r'$\mu_1=%.2f, \mu_2=%.2f$'%(mu1,mu2),fontsize=20)
#fig.tight_layout()

#plt.savefig('./pic/test3-2.png', dpi=300)

#%%
# plot 3D J(mu_1,mu_2) and ||y-yd|| and ||u||  (little slow)
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator
import numpy as np
mu1_list = np.linspace(0.05,0.45,4)
mu2_list = np.linspace(0.5,2.5,4)

dolfin_data =  [mu1_list[0], mu2_list[0], 0.12402, 0.12402, 0.00000,
                mu1_list[0], mu2_list[1], 0.00844, 0.00611, 2.32517,
                mu1_list[0], mu2_list[2], 0.21587, 0.17658, 39.29536,
                mu1_list[0], mu2_list[3], 0.79467, 0.74501, 49.65857,
                mu1_list[1], mu2_list[0], 0.11193, 0.11193, 0.00000,
                mu1_list[1], mu2_list[1], 0.01018, 0.00872, 1.45833,
                mu1_list[1], mu2_list[2], 0.25400, 0.22295, 31.04569,
                mu1_list[1], mu2_list[3], 0.85591, 0.81338, 42.53679,
                mu1_list[2], mu2_list[0], 0.08596, 0.08596, 0.00000,
                mu1_list[2], mu2_list[1], 0.00959, 0.00959, 0.00000,
                mu1_list[2], mu2_list[2], 0.22492, 0.21470, 10.22152,
                mu1_list[2], mu2_list[3], 0.73121, 0.70699, 24.22288,
                mu1_list[3], mu2_list[0], 0.04642, 0.04642, 0.00000,
                mu1_list[3], mu2_list[1], 0.00518, 0.00518, 0.00000,
                mu1_list[3], mu2_list[2], 0.12885, 0.12885, 0.00000,
                mu1_list[3], mu2_list[3], 0.41421, 0.41103, 3.18047]

dolfin_data = np.array(dolfin_data).reshape(-1,5)

mu1_rep = dolfin_data[:,0]
mu2_rep = dolfin_data[:,1]
dolfin_J = dolfin_data[:,2]
dolfin_J1 = dolfin_data[:,3]
dolfin_J2 = dolfin_data[:,4]

n=100
m=100
X = np.linspace(0.05,0.45,n)
Y = np.linspace(0.5,2.5,n)
X, Y = np.meshgrid(X, Y)
Z1 = X.copy(); Z2 = X.copy(); Z3 = X.copy()
x0 = np.linspace(0,2,2*m)
x1 = np.linspace(0,1,m)
x0, x1 = np.meshgrid(x0,x1)
x0 = x0.reshape(-1,1); x1 = x1.reshape(-1,1);
te_data = torch.from_numpy(np.hstack([x0,x1,x1,x1])).to(device)
for i in range(n):
    for j in range(n):
        mu1 = X[i,j]; mu2 = Y[i,j]
        ind = ((x0-1.5)**2 + (x1-0.5)**2 >= mu1**2).flatten()
        te_data_tmp = te_data[ind,:]
        te_data_tmp[:,2:3] = mu1
        te_data_tmp[:,3:4] = mu2
        u = pred_u(net_u, te_data_tmp).cpu().detach().numpy()
        y = pred_y(net_y, lenth, te_data_tmp).cpu().detach().numpy()
        y_d = data.yd(te_data_tmp).cpu().detach().numpy()
        area = 2-np.pi*mu1**2
        allowance = 0.5*area*((y - y_d)**2).mean()
        control = 0.5*area*(u**2).mean()
        Z1[i,j] = allowance + data.alpha*control
        Z2[i,j] = allowance
        Z3[i,j] = control
#%%    
ss = 1e-5
ss2 = 1e-3
pt_size = 40
fs = 15
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(8,7))
#ax.plot(mu1_rep, mu2_rep, dolfin_J, '.', c='r',markersize=100)
ax.scatter(mu1_rep, mu2_rep, dolfin_J+ss, c='r',s=pt_size)
ax.legend(['dolfin-adjoint'],prop={'size':fs})
surf = ax.plot_surface(X, Y, Z1, cmap='coolwarm',
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.set_xlabel(r'$\mu_1$')
ax.set_ylabel(r'$\mu_2$')
ax.set_title(r'$J(\mu_1, \mu_2)$', fontsize=fs)
plt.savefig('./pic/test3-3d2-1.png', dpi=300)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(8,7))
ax.scatter(mu1_rep, mu2_rep, dolfin_J1+ss, c='r',s=pt_size)
ax.legend(['dolfin-adjoint'],prop={'size':fs})
surf = ax.plot_surface(X, Y, Z2, cmap='coolwarm',
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.set_xlabel(r'$\mu_1$')
ax.set_ylabel(r'$\mu_2$')
ax.set_title(r'$\frac{1}{2}||y-y_d||_{L_2}^2(\mu_1, \mu_2)$', fontsize=fs)
plt.savefig('./pic/test3-3d2-2.png', dpi=300)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(8,7))
ax.scatter(mu1_rep, mu2_rep, dolfin_J2+ss2, c='r',s=pt_size)
ax.legend(['dolfin-adjoint'],prop={'size':fs})
surf = ax.plot_surface(X, Y, Z3, cmap='coolwarm',
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.set_xlabel(r'$\mu_1$')
ax.set_ylabel(r'$\mu_2$')
ax.set_title(r'$\frac{1}{2}||u||_{L_2}^2(\mu_1, \mu_2)$', fontsize=fs)
plt.savefig('./pic/test3-3d2-3.png', dpi=300)

#%%
#relative error with dolfin
n=100
mu1 = 0.4
mu2 = 2.0

y_dal = np.load('./net_save/y-dal-(%.2f,%.1f,0.00,10.00).npy'%(mu1,mu2)).flatten()
u_dal = np.load('./net_save/u-dal-(%.2f,%.1f,0.00,10.00).npy'%(mu1,mu2)).flatten()
y_pinn = np.load('./net_save/y-pinn-(%.2f,%.1f,0.00,10.00).npy'%(mu1,mu2)).flatten()
u_pinn = np.load('./net_save/u-pinn-(%.2f,%.1f,0.00,10.00).npy'%(mu1,mu2)).flatten()

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

print('%.3e'%(relative_error_l2(u, u_dal)))
print('%.3e'%(relative_error_l2(y, y_dal)))
print('%.3e'%(relative_error_l2(u_pinn, u_dal)))
print('%.3e'%(relative_error_l2(y_pinn, y_dal)))
#print(relative_error_linf(u, u_dal))
#print(relative_error_linf(y, y_dal))


#%%
# save npy file
net_y = torch.load('./net_save/test3-pinn2-ynet.pt')
net_u = torch.load('./net_save/test3-pinn2-unet.pt')
net_p = torch.load('./net_save/test3-pinn2-pnet.pt')
loss_history = torch.load('./net_save/test3-pinn2-loss.pt').tolist()
in_set.x =  torch.load('./net_save/test3-pinn2-x.pt')

mu1 = 0.25; mu2 = 2.0; ua = 0.00; ub = 10.00

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


#np.save('y-pinn2-(%.2f,%.1f,%.2f,%.2f).npy'%(mu1,mu2,ua,ub),y)
#np.save('u-pinn2-(%.2f,%.1f,%.2f,%.2f).npy'%(mu1,mu2,ua,ub),u)




#%%
# 提交前修改图(对比不同c的pinn+projction)
# difference between AONN PINN adn dolfin for fixed mu1 and mu2(with diff)
import matplotlib as mpl
mu1 = 0.3
mu2 = 2.5
ua=0
ub=10

c = 1/data.alpha
u_dal = np.load('./net_save/u-dal-(%.2f,%.1f,%.2f,%.2f).npy'%(mu1,mu2,ua,ub)).flatten()
u_pinnp = np.load('./net_save/u-pinn-proj-(%.2f,%.1f,%.2f,%.2f)-%s.npy'%(mu1,mu2,ua,ub,c)).flatten()
u_pinnpS = np.load('./net_save/u-pinn-proj-(%.2f,%.1f,%.2f,%.2f)-%s.npy'%(mu1,mu2,ua,ub,c/10)).flatten()
u_pinnpL = np.load('./net_save/u-pinn-proj-(%.2f,%.1f,%.2f,%.2f)-%s.npy'%(mu1,mu2,ua,ub,c*10)).flatten()
u_pinn = np.load('./net_save/u-pinn-kkt-(%.2f,%.1f,%.2f,%.2f).npy'%(mu1,mu2,ua,ub)).flatten()

#u_pinn = np.load('./net_save/u-pinn-(0.25,2.0,0.00,10.00).npy').flatten()
#p_np = np.load('p-dal-(0.25,2.5).npy').flatten()

n=100
num_line = 400

fs = 12
from matplotlib.ticker import FuncFormatter
fmt = lambda x, pos: '{:.3f}'.format(x)
fmt2 = lambda x, pos: '{:.1e}'.format(x)
fig, ax = plt.subplots(3,3,figsize=(14,6))
for i in range(3):
    for j in range(3):
        ax[i,j].axis('equal')
        ax[i,j].set_xlim([0,2])
        ax[i,j].set_ylim([0,1])
        ax[i,j].axis('off')
#ax = ax.transpose()

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

x0 = x0.flatten()
x1 = x1.flatten()

#norm1 = mpl.colors.Normalize(vmin=min(u_dal.min(),u.min()),\
#                             vmax=max(u_dal.max(),u.max()))
    
norm1 = mpl.colors.Normalize(vmin=u_pinnpS.min(), vmax=u_pinnpS.max())
norm1 = mpl.colors.Normalize(vmin=u_dal.min(), vmax=u_dal.max())
norm2 = mpl.colors.Normalize(vmin=abs(u_pinn-u_dal).min(), vmax=abs(u_pinn-u_dal).max())


ax000 = ax[0,0].tricontourf(x0, x1, u_dal/2, num_line, cmap='coolwarm')
ax00 = plt.tricontourf(x0, x1, u_dal, num_line, cmap='coolwarm', norm=norm1)
ax[0,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax00,ax=ax[0,0],aspect=10,format=FuncFormatter(fmt))
ax[0,0].set_title('dolfin-adjoint',fontsize=fs)

ax01 = ax[0,1].tricontourf(x0, x1, u, num_line, cmap='coolwarm', norm=norm1)
ax[0,1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax00,ax=ax[0,1],aspect=10,format=FuncFormatter(fmt))
ax[0,1].set_title('AONN',fontsize=fs)

ax02 = ax[0,2].tricontourf(x0, x1, u_pinn, num_line, cmap='coolwarm', norm=norm1)
ax[0,2].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax00,ax=ax[0,2],aspect=10,format=FuncFormatter(fmt))
ax[0,2].set_title('PINN',fontsize=fs)

# pinn+proj
ax10 = ax[1,0].tricontourf(x0, x1, u_pinnpS, num_line, cmap='coolwarm', norm=norm1)
ax[1,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax00,ax=ax[1,0],aspect=10,format=FuncFormatter(fmt))
ax[1,0].set_title('PINN+Projection(c=100)',fontsize=fs)

ax11 = ax[1,1].tricontourf(x0, x1, u_pinnp, num_line, cmap='coolwarm', norm=norm1)
ax[1,1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax00,ax=ax[1,1],aspect=10,format=FuncFormatter(fmt))
ax[1,1].set_title('PINN+Projection(c=1000)',fontsize=fs)

ax12 = ax[1,2].tricontourf(x0, x1, u_pinnpL, num_line, cmap='coolwarm', norm=norm1)
ax[1,2].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax00,ax=ax[1,2],aspect=10,format=FuncFormatter(fmt))
ax[1,2].set_title('PINN+Projection(c=10000)',fontsize=fs)


# error
ax21 = ax[2,1].tricontourf(x0, x1, abs(u_pinn-u_dal), num_line, alpha=1, cmap='Oranges', norm=norm2)
ax[2,1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax21,ax=ax[2,1],aspect=10,format=FuncFormatter(fmt2))
ax[2,1].set_title('PINN error',fontsize=fs)

ax20 = ax[2,0].tricontourf(x0, x1, abs(u-u_dal), num_line, alpha=1, cmap='Oranges', norm=norm2)
ax[2,0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax21,ax=ax[2,0],aspect=10,format=FuncFormatter(fmt2))
ax[2,0].set_title('AONN error',fontsize=fs)

ax22 = ax[2,2].tricontourf(x0, x1, abs(u_pinnp-u_dal), num_line, alpha=1, cmap='Oranges', norm=norm2)
ax[2,2].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax21,ax=ax[2,2],aspect=10,format=FuncFormatter(fmt2))
ax[2,2].set_title('PINN+Projection(c=1000) error',fontsize=fs)

#plt.suptitle(r'$\mu_1=%.2f, \mu_2=%.2f$'%(mu1,mu2),fontsize=20)
#fig.tight_layout()
plt.savefig('./pic/test3-diff-v5.png', dpi=300)


#%%
# load nxn data from dolfin and compute the u error
import numpy as np
# Error norm
def error_l2(u, ua, area):
    return (((u-ua)**2).mean()*area) ** 0.5

# Error norm
def relative_error_l2(u, ua, area):
    return (((u-ua)**2).sum() / ((ua**2).sum()+1e-16)) ** 0.5


file_path = './'
n_para = 16

mu1_list = np.linspace(0.05, 0.45, n_para)
mu2_list = np.linspace(0.5, 2.5, n_para)
error_list = np.zeros(n_para**2)
ii = 0
for mu1 in mu1_list:
    for mu2 in mu2_list:
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
        aonn_u = pred_u(net_u, te_data).cpu().detach().numpy()
        fem_u = np.load(file_path+'fem_u{}.npy'.format(ii+1))
        #print(fem_u.shape)
        error_list[ii] = error_l2(aonn_u, fem_u, area)
        ii += 1
print(error_list.mean(), error_list.std())

#%%
num_line = 100
i = 3
j = 1
mu1 = mu1_list[i]
mu2 = mu2_list[j]
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
aonn_u = pred_u(net_u, te_data).cpu().detach().numpy()
fem_u = np.load(file_path+'fem{}.npy'.format(i*4+j+1))
#fem_u = np.load(file_path+'fems.npy')

fig, ax = plt.subplots(2,1,figsize=(8,7))
for j in range(2):
    ax[j].axis('equal')
    ax[j].set_xlim([0,2])
    ax[j].set_ylim([0,1])
    ax[j].axis('off')
    
ax0 = ax[0].tricontourf(x0.flatten(), x1.flatten(), aonn_u.flatten(), num_line, cmap='coolwarm')

ax[0].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax0,ax=ax[0])
ax[0].set_title('AONN: u')

ax1 = ax[1].tricontourf(x0.flatten(), x1.flatten(), fem_u.flatten(), num_line, alpha=1, cmap='coolwarm')
ax[1].add_artist(plt.Circle((1.5,0.5), mu1, fill=True, color='white'))
fig.colorbar(ax1,ax=ax[1])
ax[1].set_title('AONN: y')


'''
#%%
# bo ge
import numpy as np
x = np.linspace(-2.0, 2.0, 200)
y = np.linspace(-2.0, 2.0, 200)
mx, my = np.meshgrid(x, y)
r=2.05
ind = ((mx**2 + my**2) <= r**2)
mx = mx[ind].flatten()
my = my[ind].flatten()
u = ((mx**2+my**2)**0.5-1.0)**2 * (2.0-(mx**2+my**2)**0.5)**2 *\
    np.sin(2*np.pi*mx)*np.sin(2*np.pi*my)
fig, ax = plt.subplots(1, 1)
c_out = plt.Circle((0,0), 2.1, color='w',lw=10, fill=False)
ax.add_artist(c_out)
ss = ax.tricontourf(mx, my, u, 100, cmap='coolwarm')
fig.colorbar(ss, ax=ax)
ax.axis('equal')
ax.set_xlim([-2.1,2.1])
ax.set_ylim([-2.1,2.1])

#%%
# bo ge
def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


fig, ax = newfig(1.0, 1.1)
gridspec.GridSpec(1,1)
ax = plt.subplot2grid((1,1), (0,0))
tcf = ax.tricontourf(triang_total, u_pred.flatten(), 100 ,cmap='jet')
ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor = 'w'))
tcbar = fig.colorbar(tcf)
tcbar.ax.tick_params(labelsize=28)
ax.set_xlabel('$x$', fontsize = 32)
ax.set_ylabel('$y$', fontsize = 32)
ax.set_title('$u$ (Predicted)', fontsize = 34)
ax.tick_params(axis="x", labelsize = 28)
ax.tick_params(axis="y", labelsize = 28)   
plt.plot(X_fi1_train_Plot[:,0:1], X_fi1_train_Plot[:,1:2], 'w-', markersize =2, label='Interface Pts')
plt.plot(X_fi2_train_Plot[:,0:1], X_fi2_train_Plot[:,1:2], 'w-', markersize =2, label='Interface Pts')
#fig.tight_layout()
fig.set_size_inches(w=12,h=9)
savefig('./figures/XPINN_PoissonEq_Sol') 
plt.show()  
#这是他的代码
'''
