from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import shutil 
import itertools

import torch
import numpy as np
from icecream import ic 
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader 
from scipy.stats import qmc

from pde_model.spaces import FiniteChebyshev
from pde_model.system import ODESystem
from nn_model import DeepONet 
import BR_lib.BR_model as BR_model


torch.manual_seed(999)
torch.cuda.manual_seed_all(999)
np.random.seed(999)


def ode_system(T):
    """ODE"""

    def g(s, u, x):
        # Antiderivative
        return u
        # Nonlinear ODE
        # return -s**2 + u
        # Gravity pendulum
        # k = 1
        # return [s[1], - k * np.sin(s[0]) + u]

    s0 = [0]
    # s0 = [0, 0]  # Gravity pendulum
    return ODESystem(g, s0, T)


def loss_func(Model, 
              SensorValues, 
              CollocationPoints, 
              SourceTerm, 
              beta=100.):
    
    num = SensorValues.shape[0] 
    m = SensorValues.shape[1] 
    Q = CollocationPoints.shape[1] 

    BoundaryPoints = torch.zeros(num, 1).to(SensorValues.device) 
    BoundaryValues = torch.zeros(num, 1).to(SensorValues.device) 

    BoundaryResidual = (Model(BoundaryPoints, SensorValues) - BoundaryValues)**2

    # ic(SensorValues.shape, Q)

    SensorValues_ = SensorValues.repeat(1,Q).reshape(-1,m)
    CollocationPoints_ = CollocationPoints.reshape(-1,1)
    SourceTerm_ = SourceTerm.reshape(-1,1) 
    
    CollocationPoints_.requires_grad = True 

    # ic(SensorValues_.shape, CollocationPoints_.shape, SourceTerm_.shape)

    s = Model(CollocationPoints_, SensorValues_) 

    s_t,  = torch.autograd.grad(
                                s, 
                                CollocationPoints_, 
                                create_graph=True, 
                                retain_graph=True, 
                                grad_outputs=torch.ones(num*Q,1).to(SensorValues.device)
                                )
    
    InnerResidual = torch.mean((((s_t - SourceTerm_).reshape(num, Q))**2), 1, True) 

    Residual = beta * InnerResidual 
    # + beta*BoundaryResidual 
    objective = Residual.mean()

    return objective, Residual 

def entropy_loss_func(Model, X, Quantity, Pre_Pdf):
    log_pdf = torch.clamp(Model(X), -23.02585, 5.0)

    # scaling for numerical stability
    scaling = 1000.0
    Pre_Pdf = scaling*Pre_Pdf
    Quantity = scaling*Quantity

    # importance sampling
    ratio = Quantity / Pre_Pdf
    res_time_logpdf = ratio*log_pdf
    entropy_loss = -torch.mean(res_time_logpdf)
    return entropy_loss


def error_func(u, ua):
    # return torch.sqrt(((u-ua)**2).mean())
    return torch.mean(torch.square(u - ua))
    # return (((u-ua)**2).sum() / (ua**2).sum()) ** 0.5


def gen_nd_ball(n_sample, n_dim):
    x_g = np.random.randn(n_sample, n_dim)
    u_number = np.random.rand(n_sample, 1)
    x_normalized = x_g / np.sqrt(np.sum(x_g**2, axis=1, keepdims=True))
    x_sample = u_number**(1/n_dim) * x_normalized
    return x_sample


def main():


    if os.path.exists('./pdf_ckpt'):
        shutil.rmtree('./pdf_ckpt')
    os.mkdir('./pdf_ckpt')

    if os.path.exists('./store_data'):
        shutil.rmtree('./store_data')
    os.mkdir('./store_data')
    


    # Computing device
    device = torch.device('cuda:0')



    # Dimension
    physical_dim = 1
    parametric_dim = 8
    parametric_ub = 1.
    sensor_dim = parametric_dim # no use


    
    # Problem setup
    T = 1
    system = ode_system(T)
    space = FiniteChebyshev(N=parametric_dim, M=parametric_ub)



    # DeepONet
    branch_layers = [sensor_dim] + 5*[50]
    trunk_layers = [physical_dim] + 5*[50]
    pde_model = DeepONet(branch_layers, trunk_layers).to(device)
    pde_optim = torch.optim.Adam(pde_model.parameters(), lr=0.0001)



    # Flow 
    xlb, xhb = -parametric_ub-0.01, parametric_ub+0.01
    n_step = 2
    n_depth = 6
    n_width = 64
    n_bins4cdf = 32
    shrink_rate = 1.0
    flow_coupling = 1
    rotation = False
    bounded_supp = False
    pdf_model = BR_model.IM_rNVP_KR_CDF(parametric_dim,
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



    # Training
    das_or_rar = 'das'
    qusi = 0
    das_mode = 'add'
    n_train = 20000 # The number of samples in the initial training set
    rar_add = 7500
    n_test = 20000
    n_epochs =  3000 
    batch_size = 5000 # Batch size of training generator
    flow_epochs = 3000 
    flow_batch_size = 5000 
    epochs_i = 100 
    Q = 100 # The number of training points of every sample 
    max_stage = 5



    difficulty = 6.



    # Test dataset 

    TestFeatures1 = 0.5 * gen_nd_ball(n_test//2, parametric_dim) + 0.5 
    TestFeatures2 = 2 * parametric_ub * np.random.rand(n_test//2, parametric_dim) - parametric_ub 
    TestFeatures = np.concatenate([TestFeatures1, TestFeatures2], axis=0)
    information_ = system.gen_operator_data(space, sensor_dim, Q, n_test, mode='test', features=TestFeatures)
    test_points = torch.tensor(information_['collocation_points'], dtype=torch.float32).to(device)
    test_sensor = torch.tensor(information_['features'], dtype=torch.float32).to(device) 

    test_singular_origin = torch.tensor(information_['features'], dtype=torch.float32).to(device) 
    test_singular_term = torch.exp( -difficulty * torch.sum((test_singular_origin-0.5)**2, 1, True) )

    test_solu = torch.tensor(information_['solution_on_collocation_points'], dtype=torch.float32).to(device)

    test_solu = test_singular_term * test_solu 



    ##########################################################
    #################### Start Training ######################
    ##########################################################
    if qusi == 1:
        sampler = qmc.Sobol(d=parametric_dim)
        manufeatures = 2 * parametric_ub * sampler.random(n_train) - parametric_ub
    else:
        manufeatures = 2 * parametric_ub * np.random.rand(n_train, parametric_dim) - parametric_ub 
    
    manufeatures_for_flow = manufeatures 

    loss_vs_iter = []
    error_vs_iter = []

    for st in range(max_stage):

        manufeatures = np.array(manufeatures)
        manufeatures_for_flow = np.array(manufeatures_for_flow)



        information = system.gen_operator_data(space, sensor_dim, Q, n_train, mode='train', features=manufeatures)
        manufeatures = torch.tensor(manufeatures, dtype=torch.float32)
        sensor_values = torch.tensor(information['features'], dtype=torch.float32)
        collocation_points = torch.tensor(information['collocation_points'], dtype=torch.float32)

        singular_origin = torch.tensor(information['features'], dtype=torch.float32)
        singular_term = torch.exp( -difficulty * torch.sum((singular_origin-0.5)**2, 1, True) )

        source_term = torch.tensor(information['source_on_collocation_points'], dtype=torch.float32)

        source_term = singular_term * source_term 



        pde_train_data_set = TensorDataset(manufeatures, sensor_values, collocation_points, source_term) 
        pde_train_data_loader = DataLoader(pde_train_data_set, shuffle=True, batch_size=batch_size) 



        for i in range(int(n_epochs/epochs_i)):
            for j in range(epochs_i):
                for _, trains in enumerate(pde_train_data_loader):

                    train_a, train_b, train_c, train_d = trains 

                    train_a = train_a.to(device)
                    train_b = train_b.to(device)
                    train_c = train_c.to(device)
                    train_d = train_d.to(device)


                    pde_optim.zero_grad()
                    # scores = deeponet(x_train, sensor_values_train)
                    # loss = F.mse_loss(scores, y_train, reduction='mean')
                    loss, _ = loss_func(pde_model, train_b, train_c, train_d) 
                    loss.backward()
                    pde_optim.step()

                    error = error_func(pde_model(test_points.reshape(-1,1), 
                                                 test_sensor.repeat(1,Q).reshape(-1,sensor_dim)).detach(),
                                       test_solu.reshape(-1,1))

                    loss_vs_iter.append(loss.item()) 
                    error_vs_iter.append(error.item())
                
                torch.save(pde_model, './pdf_ckpt/pde_model{}.pt'.format(st*n_epochs+i*epochs_i+j))
                if st*n_epochs+i*epochs_i+j >= 5:
                    os.remove('./pdf_ckpt/pde_model{}.pt'.format(st*n_epochs+i*epochs_i+j-5))


            
            ic(st, i*epochs_i+j, loss.item(), error.item())


        if das_or_rar == 'das':
            #############################################
            if st < max_stage-1:
                #############################################
                information_for_flow = system.gen_operator_data(space, sensor_dim, Q, n_train, mode='train', features=manufeatures_for_flow)
                manufeatures_for_flow = torch.tensor(manufeatures_for_flow, dtype=torch.float32)
                sensor_values_for_flow = torch.tensor(information_for_flow['features'], dtype=torch.float32)
                collocation_points_for_flow = torch.tensor(information_for_flow['collocation_points'], dtype=torch.float32)

                singular_origin_for_flow = torch.tensor(information_for_flow['features'], dtype=torch.float32)
                singular_term_for_flow = torch.exp( -difficulty * torch.sum((singular_origin_for_flow-0.5)**2, 1, True) )

                source_term_for_flow = torch.tensor(information_for_flow['source_on_collocation_points'], dtype=torch.float32)

                source_term_for_flow = singular_term_for_flow * source_term_for_flow 

                if st == 0:
                    pre_pdf = torch.ones_like(manufeatures_for_flow[:,0:1])
                else:
                    pre_pdf = torch.exp(pdf_model(manufeatures_for_flow.to(device))).detach().cpu()
                    pre_pdf = torch.clamp(pre_pdf, 1.0e-10, 148.4131)



                pde_train_data_set_for_flow = TensorDataset(manufeatures_for_flow, 
                                                            sensor_values_for_flow, 
                                                            collocation_points_for_flow, 
                                                            source_term_for_flow, 
                                                            pre_pdf) 
                pde_train_data_loader_for_flow = DataLoader(pde_train_data_set_for_flow, shuffle=True, batch_size=flow_batch_size) 

                # solve flow 
                for i in range(flow_epochs):
                    for step, batch_x in enumerate(pde_train_data_loader_for_flow):
                        flow_a, flow_b, flow_c, flow_d, flow_e = batch_x
                        flow_a = flow_a.to(device)
                        flow_b = flow_b.to(device)
                        flow_c = flow_c.to(device)
                        flow_d = flow_d.to(device)
                        flow_e = flow_e.to(device)


                        pdf_optim.zero_grad()

                        _, quantity = loss_func(pde_model, flow_b, flow_c, flow_d) 
                        cross_entropy = entropy_loss_func(pdf_model, flow_a, quantity, flow_e)

                        cross_entropy.backward(retain_graph=True) 
                        pdf_optim.step()

                    if i%100 == 0:
                        ic(st, i, step, cross_entropy.item())

                
                x_prior = pdf_model.draw_samples_from_prior(n_train, parametric_dim).to(device)
                x_candidate = pdf_model.mapping_from_prior(x_prior).detach().cpu()

                np.savetxt('./store_data/stage_{}_points.dat'.format(st+1), np.array(x_candidate))

                if das_mode == 'add':
                    manufeatures = torch.cat((manufeatures, x_candidate), 0)
                else:
                    manufeatures = x_candidate 
                
                buffersize = manufeatures.shape[0]

                x_prior = pdf_model.draw_samples_from_prior(buffersize, parametric_dim).to(device)
                manufeatures_for_flow = pdf_model.mapping_from_prior(x_prior).detach().cpu()



                # resample
        elif das_or_rar == 'rar':
            rar_features_prior = 2 * parametric_ub * np.random.rand(4*rar_add, parametric_dim) - parametric_ub
            rar_information = system.gen_operator_data(space, sensor_dim, Q, 4*rar_add, mode='train', features=rar_features_prior)
            rar_manufeatures = torch.tensor(rar_features_prior, dtype=torch.float32)
            rar_sensor_values = torch.tensor(rar_information['features'], dtype=torch.float32)
            rar_collocation_points = torch.tensor(rar_information['collocation_points'], dtype=torch.float32)

            rar_singular_origin = torch.tensor(rar_information['features'], dtype=torch.float32)
            rar_singular_term = torch.exp( -difficulty * torch.sum((rar_singular_origin-0.5)**2, 1, True) )

            rar_source_term = torch.tensor(rar_information['source_on_collocation_points'], dtype=torch.float32)

            rar_source_term = rar_singular_term * rar_source_term

            rar_sensor_values = rar_sensor_values.to(device)
            rar_collocation_points = rar_collocation_points.to(device)
            rar_source_term = rar_source_term.to(device)

            _, rar_res = loss_func(pde_model, rar_sensor_values, rar_collocation_points, rar_source_term)
            rar_res = rar_res.detach().clone().cpu().numpy()
            idx = np.squeeze((-rar_res).argsort(axis=0)[:rar_add])
            x_candidate = torch.tensor(rar_features_prior[idx], dtype=torch.float32)

            manufeatures = torch.cat((manufeatures, x_candidate), 0)

            np.savetxt('./store_data/stage_{}_points.dat'.format(st+1), np.array(x_candidate))






            
    np.savetxt('./store_data/loss_vs_iter.dat', np.array(loss_vs_iter))
    np.savetxt('./store_data/error_vs_iter.dat', np.array(error_vs_iter))


if __name__ == '__main__':
    main()