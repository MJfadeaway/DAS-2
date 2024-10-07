import torch
import numpy as np
import time
import data_pocp as data
import itertools
import matplotlib.pyplot as plt
import bfgs
import os 
import shutil

from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader 
from icecream import ic
from scipy.stats import qmc

import BR_lib.BR_model as BR_model


seed = 6
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.set_default_dtype(torch.float64)


class FCNN(torch.nn.Module):
    def __init__(self, layers):
        super(FCNN, self).__init__()
        self.layers = layers
        self.layers_hid_num = len(layers)-1

        fc = []
        for i in range(self.layers_hid_num):
            fc.append(torch.nn.Linear(self.layers[i],self.layers[i+1]))
        self.fc = torch.nn.Sequential(*fc)
    
    def forward(self, x):
        for i in range(self.layers_hid_num-1):
            x = torch.tanh(self.fc[i](x))
        
        return self.fc[-1](x)

def entropy_loss_func(Model, X, Quantity, Pre_Pdf):
    log_pdf = torch.clamp(Model(X), np.log(1.0e-8), 3.0)

    # scaling for numerical stability
    scaling = 1000.0
    Pre_Pdf = scaling*Pre_Pdf
    Quantity = scaling*Quantity

    # importance sampling
    ratio = Quantity / Pre_Pdf
    res_time_logpdf = ratio*log_pdf
    entropy_loss = -torch.mean(res_time_logpdf)
    return entropy_loss
    

def length_factor(Input):

    return Input[:,0:1] * (2-Input[:,0:1]) * Input[:,1:2] * (1-Input[:,1:2]) \
           * ( (Input[:,0:1]-1.5)**2 + (Input[:,1:2]-0.5)**2 - Input[:,2:3]**2 )

def yd(x):
    tmp = 0*x[:,0:1]
    mu2 = x[:,3:4]
    omega1 = (x[:,0:1]<=1)
    omega2 = (x[:,0:1]>1)
    tmp[omega1] = 1.    
    tmp[omega2] = mu2[omega2]
    return tmp

def approx_y(Net, Input):
    return Net(Input)*length_factor(Input) + 1

def approx_p(Net, Input):
    return Net(Input)*length_factor(Input)


def loss_func_kkt(NetY, NetP, Input):
    Input.requires_grad = True
    y = approx_y(NetY, Input)
    p = approx_p(NetP, Input)
    u = data.P( -1/data.alpha * p )
    # p = -data.alpha * tmp
    y_d = yd(Input)
    
    yx, = torch.autograd.grad(y, Input,
                              create_graph=True, retain_graph=True,
                              grad_outputs=torch.ones_like(Input[:,0:1]))
    yx0 = yx[:,0:1]
    yx1 = yx[:,1:2]
    yx0x, = torch.autograd.grad(yx0, Input,
                                create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones_like(Input[:,0:1]))
    yx0x0 = yx0x[:,0:1]
    yx1x, = torch.autograd.grad(yx1, Input,
                                create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones_like(Input[:,0:1]))
    yx1x1 = yx1x[:,1:2]

    res_y = -(yx0x0+yx1x1) - u

    px, = torch.autograd.grad(p, Input,
                              create_graph=True, retain_graph=True,
                              grad_outputs=torch.ones_like(Input[:,0:1]))
    px0 = px[:,0:1]
    px1 = px[:,1:2]
    px0x, = torch.autograd.grad(px0, Input,
                                create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones_like(Input[:,0:1]))
    px0x0 = px0x[:,0:1]
    px1x, = torch.autograd.grad(px1, Input,
                                create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones_like(Input[:,0:1]))
    px1x1 = px1x[:,1:2]
    
    res_p = -(px0x0+px1x1) - (y - y_d)
        
              
    residual = res_y**2 + res_p**2
                 
    return torch.sqrt(residual.mean()), residual

def filter_2d(Input):
    ind = (Input[:,0]-1.5)**2 + (Input[:,1]-0.5)**2 > Input[:,2]**2
    return Input[ind,:]



def relative_error(u, ua):
    return (((u-ua)**2).sum() / ((ua**2).sum()+1e-16)) ** 0.5

# -------------------------------------------------------------------------------------------------
# Hyperparameters

lb = [0., 0., 0.05, 0.5]
hb = [2., 1., 0.45, 2.5]


n_dim = len(lb)
lr = 1e-4

layers = [n_dim] + 6*[25] + [1]

qusi = 0
use_das = 1
das_mode = 'add'
n_train = 4000
n_add = n_train
batch_size = n_train
n_epochs = 2000
i_epochs = 100
max_stage = 5

optim_type = 'bfgs'

# -------------------------------------------------------------------------------------------------
# Construction

net_y = FCNN(layers).to(device)
net_p = FCNN(layers).to(device)

if optim_type == 'adam':
    pde_optim = torch.optim.Adam(itertools.chain(net_y.parameters(), net_p.parameters()),lr=lr)
elif optim_type == 'bfgs':
    pde_optim = bfgs.BFGS(itertools.chain(net_y.parameters(),
                                            net_p.parameters()),
                            lr=1, max_iter=i_epochs,
                            tolerance_grad=1e-16, tolerance_change=1e-16,
                            line_search_fn='strong_wolfe')
elif optim_type == 'lbfgs':
    pde_optim = torch.optim.LBFGS(itertools.chain(net_y.parameters(),
                                                    net_p.parameters()),
                                    lr=1, max_iter=i_epochs,
                                    tolerance_grad=1e-16, tolerance_change=1e-16,
                                    line_search_fn='strong_wolfe')



xlb = [0.-0.01, 0.-0.005, 0.05-0.002, 0.5-0.01]
xhb = [2.+0.01, 1.+0.005, 0.45+0.002, 2.5+0.01]
n_step = 2
n_depth = 6
n_width = 24
n_bins4cdf = 32
shrink_rate = 1.0
flow_coupling = 1
rotation = False
bounded_supp = False
pdf_model = BR_model.IM_rNVP_KR_CDF(n_dim,
                                    xlb, xhb,
                                    n_step,
                                    n_depth,
                                    n_width = n_width,
                                    n_bins=n_bins4cdf,
                                    shrink_rate=shrink_rate,
                                    flow_coupling=flow_coupling,
                                    rotation=rotation,
                                    bounded_supp=bounded_supp).to(device)
pdf_optim = torch.optim.Adam(pdf_model.parameters(),lr=lr)


if qusi == 0:
    pde_feature = torch.zeros(3*n_train,0)
    for di in range(n_dim):
        tmp = lb[di] + (hb[di]-lb[di])*torch.rand(3*n_train,1)
        pde_feature = torch.cat([pde_feature, tmp], 1)
else:
    sampler = qmc.Sobol(d=n_dim, seed=seed)
    pde_feature = torch.tensor(lb) + (torch.tensor(hb) - torch.tensor(lb)) * sampler.random(n=3*n_train)

# pde_feature = pde_feature.to(torch.float32)


pde_feature = filter_2d(pde_feature)[:n_train,:]
pdf_feature = pde_feature.detach().clone()


valid_x = np.load('./valid_data/valid_input.npy')
valid_u = np.load('./valid_data/valid_output.npy')
valid_x = torch.from_numpy(valid_x).to(device)
# valid_u = torch.from_numpy(valid_u).to(device)

# -------------------------------------------------------------------------------------------------


if os.path.exists('./pdf_ckpt'):
    shutil.rmtree('./pdf_ckpt')
os.mkdir('./pdf_ckpt')

if os.path.exists('./store_data'):
    shutil.rmtree('./store_data')
os.mkdir('./store_data')

# np.savetxt('./store_data/stage_0_points.dat', np.array(pde_feature))

err_history = []
loss_history = []

for st in range(max_stage):

    if optim_type == 'lbfgs':
        pde_optim = torch.optim.LBFGS(itertools.chain(net_y.parameters(),
                                                        net_p.parameters()),
                                        lr=1, max_iter=i_epochs,
                                        tolerance_grad=1e-16, tolerance_change=1e-16,
                                        line_search_fn='strong_wolfe')


    if st == max_stage-1 or optim_type == 'bfgs':
        pde_optim = bfgs.BFGS(itertools.chain(net_y.parameters(),
                                                net_p.parameters()),
                                lr=1, max_iter=i_epochs,
                                tolerance_grad=1e-16, tolerance_change=1e-16,
                                line_search_fn='strong_wolfe')


    if optim_type == 'adam':

        # Training data set 
        pde_train_data_set = TensorDataset(pde_feature)
        pde_train_data_loader = DataLoader(pde_train_data_set, shuffle=True, batch_size=batch_size)


        for i in range(n_epochs):


            for _, trains in enumerate(pde_train_data_loader):

                train_feature = trains[0]
                train_feature = train_feature.to(device)

                pde_optim.zero_grad()
                loss, _ = loss_func_kkt(net_y, net_p, train_feature)
                loss.backward()
                pde_optim.step()

            if i%i_epochs == 0:
                pred_u = data.P( -1/data.alpha * approx_p(net_p, valid_x) ).cpu().detach().clone().numpy()
                err = relative_error(pred_u, valid_u)
                ic(st+1, i+1, loss.item(), err.item())
                loss_history.append(loss.item())
                err_history.append(err.item())
    else:
        train_feature = pde_feature.to(device)

        for i in range(n_epochs//i_epochs):
            def closure():
                pde_optim.zero_grad()
                loss, _ = loss_func_kkt(net_y, net_p, train_feature)
                loss.backward()
                return loss
            pde_optim.step(closure)
        
            loss, _ = loss_func_kkt(net_y, net_p, train_feature)
            pred_u = data.P( -1/data.alpha * approx_p(net_p, valid_x) ).cpu().detach().clone().numpy()
            err = relative_error(pred_u, valid_u)
            ic(st+1, (i+1)*i_epochs, loss.item(), err.item())
            loss_history.append(loss.item())
            err_history.append(err.item())
            

    
    if use_das == 1:

        if st < max_stage-1:


            if st == 0:
                pre_pdf = torch.ones_like(pdf_feature[:,0:1])
            else:
                pre_pdf = torch.exp(pdf_model(pdf_feature.to(device))).detach().cpu()
                pre_pdf = torch.clamp(pre_pdf, 1.0e-8, np.exp(3.0))

            pde_train_data_set_for_flow = TensorDataset(pdf_feature, 
                                                        pre_pdf) 
            pde_train_data_loader_for_flow = DataLoader(pde_train_data_set_for_flow, shuffle=True, batch_size=batch_size)

            # solve flow 
            for i in range(n_epochs):
                for step, batch_x in enumerate(pde_train_data_loader_for_flow):
                    flow_a, flow_b = batch_x
                    flow_a = flow_a.to(device)
                    flow_b = flow_b.to(device)


                    pdf_optim.zero_grad()

                    _, quantity = loss_func_kkt(net_y, net_p, flow_a) 
                    cross_entropy = entropy_loss_func(pdf_model, flow_a, quantity, flow_b)

                    cross_entropy.backward(retain_graph=True) 
                    pdf_optim.step()

                if i%100 == 0:
                    ic(st+1, i, cross_entropy.item()) 


            if das_mode == 'add':
                x_prior = pdf_model.draw_samples_from_prior(3*n_add, n_dim).to(device)
                x_candidate = pdf_model.mapping_from_prior(x_prior).detach().cpu()

                x_candidate = filter_2d(x_candidate)[:n_add,:]

                pde_feature = torch.cat((pde_feature, x_candidate), 0)
                buffersize = pde_feature.shape[0]

                x_prior = pdf_model.draw_samples_from_prior(3*buffersize, n_dim).to(device)
                pdf_feature = pdf_model.mapping_from_prior(x_prior).detach().cpu()
                pdf_feature = filter_2d(pdf_feature)[:buffersize,:]
            else:
                x_prior = pdf_model.draw_samples_from_prior(3*n_train, n_dim).to(device)
                x_candidate = pdf_model.mapping_from_prior(x_prior).detach().cpu()

                x_candidate = filter_2d(x_candidate)[:n_train,:]

                pde_feature = x_candidate 
                pdf_feature = x_candidate.detach().clone()
            
            np.savetxt('./store_data/stage_{}_points.dat'.format(st+1), np.array(x_candidate))

torch.save(net_y, './pdf_ckpt/pocp_net_y.pt')
torch.save(net_p, './pdf_ckpt/pocp_net_p.pt')

np.savetxt('./store_data/err_history.dat', np.array(err_history))
np.savetxt('./store_data/loss_history.dat', np.array(loss_history))
