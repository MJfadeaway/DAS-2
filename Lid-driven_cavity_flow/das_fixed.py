# ----------------------------------------------------------------------------------------------------
# Least-Squares Neural Network Method
# ----------------------------------------------------------------------------------------------------

import argparse
import torch
import numpy as np
import time
import data
import torch.nn as nn
import BR_lib.BR_model as BR_model
import os
import shutil
import bfgs

from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader 
from icecream import ic


# ----------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------
def entropy_loss_func(Model, X, Quantity, Pre_Pdf):
    log_pdf = torch.clamp(Model(X), np.log(1.0e-10), 3.0)

    # scaling for numerical stability
    scaling = 1000.0
    Pre_Pdf = scaling*Pre_Pdf
    Quantity = scaling*Quantity

    # importance sampling
    ratio = Quantity / Pre_Pdf
    res_time_logpdf = ratio*log_pdf
    entropy_loss = -torch.mean(res_time_logpdf)
    return entropy_loss


def loss_func(Net, InSet, BdSet, Re, beta):
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
    InSet.u = Net(InSet.x)
    InSet.u0 = InSet.u[:,0:1]
    InSet.u1 = InSet.u[:,1:2]
    InSet.p = InSet.u[:,2:3]
    
    InSet.u0x, = torch.autograd.grad(InSet.u0, InSet.x,
                                     create_graph=True, retain_graph=True,
                                     grad_outputs=InSet.weight_grad)
    InSet.u0x0 = InSet.u0x[:,0:1]
    InSet.u0x1 = InSet.u0x[:,1:2]
    InSet.u0x0x, = torch.autograd.grad(InSet.u0x0, InSet.x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=InSet.weight_grad)
    InSet.u0x0x0 = InSet.u0x0x[:,0:1]
    InSet.u0x1x, = torch.autograd.grad(InSet.u0x1, InSet.x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=InSet.weight_grad)
    InSet.u0x1x1 = InSet.u0x1x[:,1:2]
    
    InSet.u1x, = torch.autograd.grad(InSet.u1, InSet.x,
                                     create_graph=True, retain_graph=True,
                                     grad_outputs=InSet.weight_grad)
    InSet.u1x0 = InSet.u1x[:,0:1]
    InSet.u1x1 = InSet.u1x[:,1:2]
    InSet.u1x0x, = torch.autograd.grad(InSet.u1x0, InSet.x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=InSet.weight_grad)
    InSet.u1x0x0 = InSet.u1x0x[:,0:1]
    InSet.u1x1x, = torch.autograd.grad(InSet.u1x1, InSet.x,
                                       create_graph=True, retain_graph=True,
                                       grad_outputs=InSet.weight_grad)
    InSet.u1x1x1 = InSet.u1x1x[:,1:2]

    InSet.px, = torch.autograd.grad(InSet.p, InSet.x,
                                    create_graph=True, retain_graph=True,
                                    grad_outputs=InSet.weight_grad)
    InSet.px0 = InSet.px[:,0:1]
    InSet.px1 = InSet.px[:,1:2]

    InSet.res0 = -1/Re * (InSet.u0x0x0+InSet.u0x1x1) + \
                 InSet.u0*InSet.u0x0 + InSet.u1*InSet.u0x1 + \
                 InSet.px0
    InSet.res1 = -1/Re * (InSet.u1x0x0+InSet.u1x1x1) + \
                 InSet.u0*InSet.u1x0 + InSet.u1*InSet.u1x1 + \
                 InSet.px1
    InSet.res2 = InSet.u0x0 + InSet.u1x1

    BdSet.u = Net(BdSet.x)
    BdSet.u0 = BdSet.u[:,0:1]
    BdSet.u1 = BdSet.u[:,1:2]
    BdSet.res0 = BdSet.u0 - BdSet.r0
    BdSet.res1 = BdSet.u1 - BdSet.r1

    loss = (InSet.res0**2).mean() + (InSet.res1**2).mean() + (InSet.res2**2).mean() + \
           beta*(BdSet.weighted0 * BdSet.res0**2).mean() + beta*(BdSet.weighted1 * BdSet.res1**2).mean()
    return loss**0.5, InSet.res0**2+InSet.res1**2+InSet.res2**2

def absolute_error(u, ua):
    return max(torch.abs(u-ua))

def filler_2d(x):
    ind = (x[:,0] < 1.).float() * (x[:,0] > 0.).float() * (x[:,1] < 1.).float() * (x[:,1] > 0.).float()
    ind = ind.bool()
    return x[ind]

# ----------------------------------------------------------------------------------------------------
def train(Net, InSet, BdSet, TeSet, Re, beta, Optim, optim_type, epochs, epochs_i, batch_size): 
    """ Train neural network g

    parameters
    Net: network
    InSet: training set on the interior of domain
    BdSet: training set on the boundary of domain
    TeSet: test set
    Re: Renold number
    beta: penalty coefficient
    Optim: optimizer for training network
    optim_type: type of optimizer
    epochs: number of iteration
    epochs_i: number of inner iteration
    """
    print('Train Neural Network')
    
    """ Record the evolution history of error """
    loss_optimal, _ = loss_func(Net, InSet, BdSet, Re, beta)
    loss_optimal = loss_optimal.data
    TeSet.Ghia_I_u = Net(TeSet.Ghia_I_x).data
    TeSet.Ghia_II_u = Net(TeSet.Ghia_II_x).data
    TeSet.Ghia_I_u0 = TeSet.Ghia_I_u[:,0:1]
    TeSet.Ghia_II_u1 = TeSet.Ghia_II_u[:,1:2]
    
    error_history = torch.zeros(int(epochs/epochs_i)+1,2)
    error_history[0,0] = absolute_error(TeSet.Ghia_I_u0, TeSet.Ghia_I_u0a).data
    error_history[0,1] = absolute_error(TeSet.Ghia_II_u1, TeSet.Ghia_II_u1a).data
    torch.save(Net.state_dict(), './optimal_state/optimal_state.tar')

    print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e' 
          %(0, loss_optimal, error_history[0,0], error_history[0,1]))
    
    """ Training cycle """
    pde_feature = InSet.x.detach().clone().cpu()
    pde_train_data_set = TensorDataset(pde_feature)
    pde_train_data_loader = DataLoader(pde_train_data_set, shuffle=True, batch_size=batch_size)
    for it in range(int(epochs/epochs_i)):
        start_time = time.time()

        """ Forward and backward propogation """
        if optim_type=='adam':
            for it_i in range(epochs_i):
                for _, trains in enumerate(pde_train_data_loader):
                    train_a = trains[0]
                    InSet_i = data.InSet(None, None, device=InSet.x.device, from_out=train_a)
                    Optim.zero_grad()
                    loss, _ = loss_func(Net, InSet_i, BdSet, Re, beta)
                    loss.backward()
                    Optim.step()

        else:
            def closure():
                Optim.zero_grad()
                loss, _ = loss_func(Net, InSet, BdSet, Re, beta)
                loss.backward()
                return loss
            Optim.step(closure)
        
        """ Record the optimal parameters """
        loss, _ = loss_func(Net, InSet, BdSet, Re, beta)
        loss = loss.data

        TeSet.Ghia_I_u = Net(TeSet.Ghia_I_x).data
        TeSet.Ghia_II_u = Net(TeSet.Ghia_II_x).data
        TeSet.Ghia_I_u0 = TeSet.Ghia_I_u[:,0:1]
        TeSet.Ghia_II_u1 = TeSet.Ghia_II_u[:,1:2]
        error_history[it+1,0] = absolute_error(TeSet.Ghia_I_u0, TeSet.Ghia_I_u0a).data
        error_history[it+1,1] = absolute_error(TeSet.Ghia_II_u1, TeSet.Ghia_II_u1a).data
        if loss < loss_optimal:
            loss_optimal = loss
            torch.save(Net.state_dict(), './optimal_state/optimal_state.tar')
        
        """ Print """
        elapsed = time.time() - start_time
        print('epoch: %d, loss: %.3e, error_u0: %.3e, error_u1: %.3e, time: %.2f'
              %((it+1)*epochs_i, loss, error_history[it+1,0], error_history[it+1,1], elapsed))

    np.savetxt('./results/error_history_ls_'+optim_type+'_'+str(beta)+'.txt', error_history)

# ----------------------------------------------------------------------------------------------------
def main():
    
    """ Configurations """
    parser = argparse.ArgumentParser(description='Least-Squares Neural Network Method')
    parser.add_argument('--Re', type=float, default=100,
                        help='Reynolds number')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='device')
    parser.add_argument('--dtype', type=str, default='float64',
                        help='data type')
    parser.add_argument('--bounds', type=float, default=[0.,1.,0.,1.],
                        help='lower and upper bounds of the domain')
    parser.add_argument("--n_dim", type=int, default=2, 
                        help='The number of random dimension.')
    parser.add_argument('--seed', type=int, default=3407,
                        help='random seed')
    parser.add_argument('--nx_te', type=int, default=[100,100], # no use
                        help='size of the test set')
    parser.add_argument('--tests_num', type=int, default=1,
                        help='number of independent tests')
    
    parser.add_argument('--n_train', type=int, default=200,
                        help='size of the interior set')
    parser.add_argument('--nx_bd', type=int, default=[100,100],
                        help='size of the boundary set')
    parser.add_argument('--gp_num', type=int, default=1,
                        help='number of gaussian points')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='penalty coefficient')
    parser.add_argument('--layers', type=int, default=[2,20,20,20,20,20,3],
                        help='network structure')
    parser.add_argument('--optim_type', type=str, default='bfgs',
                        help='opimizer type')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of iterations')
    parser.add_argument('--epochs_i', type=int, default=100,
                        help='number of inner iterations')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--use_das', type=bool, default=True,
                        help='use das or not')
    parser.add_argument('--max_stage',type=int, default=5, 
                        help='Start refine samples after every stage.')
    # parser.add_argument("--n_add", type=int, default=256, 
    #                     help='The number of samples added in training set.')
    parser.add_argument('--flow_epochs',type=int, default=3000, 
                        help='Total number of training epochs for flow.')
    parser.add_argument('--batch_size',type=int, default=100, 
                        help='Batchsize for flow.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.dtype=='float16':
        torch.set_default_dtype(torch.float16)
    if args.dtype=='float32':
        torch.set_default_dtype(torch.float32)
    if args.dtype=='float64':
        torch.set_default_dtype(torch.float64)

    if os.path.exists('./optimal_state'):
        shutil.rmtree('./optimal_state')
    os.mkdir('./optimal_state')

    if os.path.exists('./store_data'):
        shutil.rmtree('./store_data')
    os.mkdir('./store_data')

    if os.path.exists('./results'):
        shutil.rmtree('./results')
    os.mkdir('./results')

    bounds = torch.tensor(args.bounds).reshape(2,2).to(device)
    nx_train = torch.tensor(args.n_train).int().to(device)
    nx_bd = torch.tensor(args.nx_bd).int().to(device)
    nx_te = torch.tensor(args.nx_te).int().to(device)

    # ------------------------------------------------------------------------------------------------
    errors = torch.zeros(args.tests_num,2)
    for it in range(args.tests_num):

        """ Generate data set """
        in_set = data.InSet(bounds, nx_train, device=device)
        bd_set = data.BdSet(bounds, nx_bd, args.gp_num, args.Re, device)
        te_set = data.TeSet(bounds, nx_te, args.Re, device)

        pdf_feature = in_set.x.detach().clone().cpu()
        
        """ Construct neural network """
        net = Net(args.layers).to(device)
        if args.optim_type=='adam':
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        if args.optim_type=='lbfgs':
            optim = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=args.epochs_i,
                                      tolerance_grad=1e-16, tolerance_change=1e-16,
                                      line_search_fn='strong_wolfe')
            

        xlb = [0.-0.01, 0.-0.01]
        xhb = [1.+0.01, 1.+0.01]
        n_step = 1
        n_depth = 6
        n_width = 24
        n_bins4cdf = 32
        shrink_rate = 1.0
        flow_coupling = 1
        rotation = False
        bounded_supp = False
        pdf_model = BR_model.IM_rNVP_KR_CDF(args.n_dim,
                                            xlb, xhb,
                                            n_step,
                                            n_depth,
                                            n_width = n_width,
                                            n_bins=n_bins4cdf,
                                            shrink_rate=shrink_rate,
                                            flow_coupling=flow_coupling,
                                            rotation=rotation,
                                            bounded_supp=bounded_supp).to(device)
        pdf_optim = torch.optim.Adam(pdf_model.parameters(),lr=args.lr)

        """ Training """
        for ap_it in range(args.max_stage):
            if args.optim_type=='bfgs':
                optim = bfgs.BFGS(net.parameters(), lr=1, max_iter=args.epochs_i,
                                        tolerance_grad=1e-16, tolerance_change=1e-16,
                                        line_search_fn='strong_wolfe')
            start_time = time.time()
            train(net, in_set, bd_set, te_set, args.Re, args.beta, optim, args.optim_type,
                args.epochs, args.epochs_i, args.batch_size)
            elapsed = time.time() - start_time
            print('train time: %.2f' %(elapsed))

            if args.use_das:

                if ap_it < args.max_stage-1:
                    if ap_it == 0:
                        pre_pdf = torch.ones_like(pdf_feature[:,0:1])
                    else:
                        pre_pdf = torch.exp(pdf_model(pdf_feature.to(device))).detach().clone().cpu()
                        pre_pdf = torch.clamp(pre_pdf, 1.0e-10, np.exp(3.0))

                    pde_train_data_set_for_flow = TensorDataset(pdf_feature, 
                                                                pre_pdf) 
                    pde_train_data_loader_for_flow = DataLoader(pde_train_data_set_for_flow, shuffle=True, batch_size=args.batch_size)

                    # solve flow 
                    for i in range(args.flow_epochs):
                        for step, batch_x in enumerate(pde_train_data_loader_for_flow):
                            flow_a, flow_b = batch_x
                            flow_a = flow_a.to(device)
                            flow_b = flow_b.to(device)

                            flow_a_set = data.InSet(None, None, device, from_out=flow_a)
                            pdf_optim.zero_grad()

                            _, quantity = loss_func(net, flow_a_set, bd_set, args.Re, args.beta) 
                            cross_entropy = entropy_loss_func(pdf_model, flow_a, quantity, flow_b)

                            cross_entropy.backward(retain_graph=True) 
                            pdf_optim.step()

                        if i%100 == 0:
                            ic(ap_it+1, i, cross_entropy.item())
                    
                    x_prior = pdf_model.draw_samples_from_prior(3*args.n_train, args.n_dim).to(device)
                    x_candidate = filler_2d(pdf_model.mapping_from_prior(x_prior).detach().clone().cpu())[:args.n_train,:]


                    pde_feature = torch.cat((in_set.x.detach().clone().cpu(), x_candidate), 0)
                    in_set = data.InSet(None, None, device, from_out=pde_feature)

                    buffersize = pde_feature.shape[0]
                    x_prior = pdf_model.draw_samples_from_prior(3*buffersize, args.n_dim).to(device)
                    pdf_feature = filler_2d(pdf_model.mapping_from_prior(x_prior).detach().clone().cpu())[:buffersize,:]

                    np.savetxt('./store_data/stage_{}_points.dat'.format(ap_it+1), np.array(x_candidate))



        """ Inference """
        net.load_state_dict(torch.load('./optimal_state/optimal_state.tar'))
        te_set.Ghia_I_u = net(te_set.Ghia_I_x).data
        te_set.Ghia_II_u = net(te_set.Ghia_II_x).data
        te_set.Ghia_I_u0 = te_set.Ghia_I_u[:,0:1]
        te_set.Ghia_II_u1 = te_set.Ghia_II_u[:,1:2]
        
        errors[it,0] = absolute_error(te_set.Ghia_I_u0, te_set.Ghia_I_u0a).data
        errors[it,1] = absolute_error(te_set.Ghia_II_u1, te_set.Ghia_II_u1a).data
        print('error_u0: %.3e, error_u1: %.3e\n' %(errors[it,0], errors[it,1]))

    # ------------------------------------------------------------------------------------------------
    print(errors.data)
    errors_mean = errors.mean(0)
    errors_std = errors.std(0)
    print('error_u0_mean = %.3e, error_u0_std = %.3e' %(errors_mean[0], errors_std[0]))
    print('error_u1_mean = %.3e, error_u1_std = %.3e' %(errors_mean[1], errors_std[1]))

if __name__ == '__main__':
    main()
