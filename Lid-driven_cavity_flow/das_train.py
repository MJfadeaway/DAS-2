from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os 
import shutil
import time


import torch
import numpy as np
import scipy.io as scio
import torch.nn as nn
import bfgs
import BR_lib.BR_model as BR_model
import data


from icecream import ic 
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader 
from pyDOE import lhs



seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.set_default_dtype(torch.float64)

class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a


def loss_func(Net, Inner_x, BdSet, beta=1.):
    """ loss function for training network g

    parameters
    Net: network
    InSet: training set on the interior of domain
    BdSet: training set on the boundary of domain
    Re: Renold number
    beta: penalty coefficient

    returns
    loss: value of loss function
    """
    Inner_x.requires_grad = True
    Inner_weight_grad = torch.ones_like(Inner_x[:,0:1])
    Re = 100 + (1000-100) * Inner_x[:,2:3]

    Inner_u = Net(Inner_x)
    Inner_u0 = Inner_u[:,0:1]
    Inner_u1 = Inner_u[:,1:2]
    Inner_p = Inner_u[:,2:3]
    
    Inner_u0x, = torch.autograd.grad(Inner_u0, Inner_x,
                                     create_graph=True, retain_graph=True,
                                     grad_outputs=Inner_weight_grad)
    Inner_u0x0 = Inner_u0x[:,0:1]
    Inner_u0x1 = Inner_u0x[:,1:2]
    Inner_u0x0x, = torch.autograd.grad(Inner_u0x0, Inner_x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=Inner_weight_grad)
    Inner_u0x0x0 = Inner_u0x0x[:,0:1]
    Inner_u0x1x, = torch.autograd.grad(Inner_u0x1, Inner_x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=Inner_weight_grad)
    Inner_u0x1x1 = Inner_u0x1x[:,1:2]
    
    Inner_u1x, = torch.autograd.grad(Inner_u1, Inner_x,
                                     create_graph=True, retain_graph=True,
                                     grad_outputs=Inner_weight_grad)
    Inner_u1x0 = Inner_u1x[:,0:1]
    Inner_u1x1 = Inner_u1x[:,1:2]
    Inner_u1x0x, = torch.autograd.grad(Inner_u1x0, Inner_x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=Inner_weight_grad)
    Inner_u1x0x0 = Inner_u1x0x[:,0:1]
    Inner_u1x1x, = torch.autograd.grad(Inner_u1x1, Inner_x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=Inner_weight_grad)
    Inner_u1x1x1 = Inner_u1x1x[:,1:2]

    Inner_px, = torch.autograd.grad(Inner_p, Inner_x,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=Inner_weight_grad)
    Inner_px0 = Inner_px[:,0:1]
    Inner_px1 = Inner_px[:,1:2]

    Inner_res0 = -1/Re * (Inner_u0x0x0+Inner_u0x1x1) + \
                 Inner_u0*Inner_u0x0 + Inner_u1*Inner_u0x1 + \
                 Inner_px0
    Inner_res1 = -1/Re * (Inner_u1x0x0+Inner_u1x1x1) + \
                 Inner_u0*Inner_u1x0 + Inner_u1*Inner_u1x1 + \
                 Inner_px1
    Inner_res2 = Inner_u0x0 + Inner_u1x1

    BdSet.u = Net(BdSet.x)
    BdSet.u0 = BdSet.u[:,0:1]
    BdSet.u1 = BdSet.u[:,1:2]
    BdSet.res0 = BdSet.u0 - BdSet.r0
    BdSet.res1 = BdSet.u1 - BdSet.r1

    loss = (Inner_res0**2).mean() + (Inner_res1**2).mean() + (Inner_res2**2).mean() + \
           beta*(BdSet.weighted0 * BdSet.res0**2).mean() + beta*(BdSet.weighted1 * BdSet.res1**2).mean()
    return loss**0.5, Inner_res0**2+Inner_res1**2+Inner_res2**2


def entropy_loss_func(Model, X, Quantity, Pre_Pdf):
    log_pdf = torch.clamp(Model(X), np.log(1.0e-8), 5.0)

    # scaling for numerical stability
    scaling = 1000.0
    Pre_Pdf = scaling*Pre_Pdf
    Quantity = scaling*Quantity

    # importance sampling
    ratio = Quantity / Pre_Pdf
    res_time_logpdf = ratio*log_pdf
    entropy_loss = -torch.mean(res_time_logpdf)
    return entropy_loss


def relative_error(u, ua):
    return max(torch.abs(u-ua).data)

def filer_3d(x):
    ind = (x[:,0] < 1.).float() * (x[:,0] > 0.).float() * (x[:,1] < 1.).float() * (x[:,1] > 0.).float() * (x[:,2] <= 1.).float() * (x[:,2] >= 0.).float()
    ind = ind.bool()
    return x[ind]



def main():



    if os.path.exists('./pdf_ckpt'):
        shutil.rmtree('./pdf_ckpt')
    os.mkdir('./pdf_ckpt')

    if os.path.exists('./store_data'):
        shutil.rmtree('./store_data')
    os.mkdir('./store_data')



    # Computing device
    device = torch.device('cuda:0') 



    # das use
    use_das = 1
    das_mode = 'add'



    # Dimension
    n_dim = 3



    # DeepONet
    layers = [n_dim] + 5*[32] + [3]
    pde_model = Net(layers).to(device)



    # Flow 
    xlb = [0-0.01,  0-0.01, 0-0.01]
    xhb = [1+0.01,  1+0.01, 1+0.01]
    n_step = 1
    n_depth = 6
    n_width = 32
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
    pdf_optim = torch.optim.Adam(pdf_model.parameters(),lr=0.0001)



    # Test data set 
    te_set = data.TeSet(bounds=None, nx=None, Re=1000, device=device)
    te_set_x_i = torch.cat( [te_set.Ghia_I_x, torch.ones_like(te_set.Ghia_I_x[:,0:1])], 1 )
    te_set_x_ii = torch.cat( [te_set.Ghia_II_x, torch.ones_like(te_set.Ghia_II_x[:,0:1])], 1)

    te_set_u0_i = te_set.Ghia_I_u0a
    te_set_u1_ii = te_set.Ghia_II_u1a

    # Boundary set
    bounds = torch.tensor([[0.,1.],
                           [0.,1.],
                           [0.,1.]]).to(device)
    nx_bd = torch.tensor([128,128,128]).int().to(device)
    bd_set = data.extended_BdSet(bounds, nx_bd, gp_num=1, Re=None, device=device)
    ic(bd_set.x.shape)




    # Training
    n_train = 10000
    n_add = n_train
    n_epochs = 10000
    i_epochs = 100
    flow_epochs = 5000
    flow_batch_size = 5000
    max_stage = 10

    lb = np.array([0.,0.,0.])
    ub = np.array([1.,1.,1.])

    pde_feature = torch.from_numpy(lb + (ub - lb) * lhs(3, n_train))
    pdf_feature = pde_feature.detach().clone()

    for st in range(max_stage):

        # Training data set 
        pde_optim = bfgs.BFGS(pde_model.parameters(), lr=1, max_iter=i_epochs,
                                        tolerance_grad=1e-16, tolerance_change=1e-16,
                                        line_search_fn='strong_wolfe')
        train_feature = pde_feature.to(device)

        for i in range(n_epochs//i_epochs):


            def closure():
                pde_optim.zero_grad()
                loss, _ = loss_func(pde_model, train_feature, bd_set)
                loss.backward()
                return loss
            pde_optim.step(closure)

            loss, _ = loss_func(pde_model, train_feature, bd_set)
            err_u0 = relative_error(pde_model(te_set_x_i)[:,0:1], te_set_u0_i)
            err_u1 = relative_error(pde_model(te_set_x_ii)[:,1:2], te_set_u1_ii)
            ic(st+1, i, loss.item(), err_u0, err_u1)


        if use_das == 1:

            if st < max_stage-1:
 

                if st == 0:
                    pre_pdf = torch.ones_like(pdf_feature[:,0:1])
                else:
                    pre_pdf = torch.exp(pdf_model(pdf_feature.to(device))).detach().cpu()
                    pre_pdf = torch.clamp(pre_pdf, 1.0e-8, np.exp(5.0))

                pde_train_data_set_for_flow = TensorDataset(pdf_feature, 
                                                            pre_pdf) 
                pde_train_data_loader_for_flow = DataLoader(pde_train_data_set_for_flow, shuffle=True, batch_size=flow_batch_size)

                # solve flow 
                for i in range(flow_epochs):
                    for step, batch_x in enumerate(pde_train_data_loader_for_flow):
                        flow_a, flow_b = batch_x
                        flow_a = flow_a.to(device)
                        flow_b = flow_b.to(device)


                        pdf_optim.zero_grad()

                        _, quantity = loss_func(pde_model, flow_a, bd_set) 
                        cross_entropy = entropy_loss_func(pdf_model, flow_a, quantity, flow_b)

                        cross_entropy.backward(retain_graph=True) 
                        pdf_optim.step()

                    if i%100 == 0:
                        ic(st+1, i, cross_entropy.item()) 


                if das_mode == 'add':
                    x_prior = pdf_model.draw_samples_from_prior(3*n_add, n_dim).to(device)
                    x_candidate = pdf_model.mapping_from_prior(x_prior).detach().cpu()
                    x_candidate = filer_3d(x_candidate)[:n_add, :]

                    pde_feature = torch.cat((pde_feature, x_candidate), 0)
                    buffersize = pde_feature.shape[0]

                    x_prior = pdf_model.draw_samples_from_prior(3*buffersize, n_dim).to(device)
                    pdf_feature = pdf_model.mapping_from_prior(x_prior).detach().cpu()
                    pdf_feature = filer_3d(pdf_feature)[:buffersize, :]
                # else:
                #     x_prior = pdf_model.draw_samples_from_prior(n_train, n_dim).to(device)
                #     x_candidate = pdf_model.mapping_from_prior(x_prior).detach().cpu()

                #     pde_feature = x_candidate 
                #     pdf_feature = x_candidate.detach().clone()
                
                np.savetxt('./store_data/stage_{}_points.dat'.format(st+1), np.array(x_candidate))

    torch.save(pde_model, './pdf_ckpt/surrogate.pt')


if __name__ == '__main__':
    main()