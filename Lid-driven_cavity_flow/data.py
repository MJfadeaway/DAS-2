# ----------------------------------------------------------------------------------------------------
# Generate Data
# ----------------------------------------------------------------------------------------------------

import torch
import numpy as np

from pyDOE import lhs

# ----------------------------------------------------------------------------------------------------
def uu(x, ind, Re):
    """ true solution
    
    parameters
    x: coordinates of data points
    ind: index of variable
    Re: Renold number

    returns
    u: values of true solution on data points
    """
    

    if ind==0:
        tmp = torch.zeros_like(x[:,0:1])
        top_ind = (torch.abs(x[:,1]-1.)<1e-8)
        tmp[top_ind] = 1.
        return tmp
    if ind==1:
        return torch.zeros_like(x[:,0:1])

# def rr(x, Re):
#     """ right hand side of gorvening equation
    
#     parameters
#     x: coordinates of data points
#     Re: Renold number
    
#     returns
#     values of right hide side on data points
#     """
#     x.requires_grad = True
#     weight = torch.ones(x.shape[0],1)

#     u0 = uu(x, 0)
#     u0x, = torch.autograd.grad(u0, x, create_graph=True, retain_graph=True,
#                                grad_outputs=weight)
#     u0x0 = u0x[:,0:1]
#     u0x1 = u0x[:,1:2]
#     u0x0x, = torch.autograd.grad(u0x0, x, create_graph=True, retain_graph=True,
#                                  grad_outputs=weight)
#     u0x0x0 = u0x0x[:,0:1]
#     u0x1x, = torch.autograd.grad(u0x1, x, create_graph=True, retain_graph=True,
#                                  grad_outputs=weight)
#     u0x1x1 = u0x1x[:,1:2]

#     u1 = uu(x, 1)
#     u1x, = torch.autograd.grad(u1, x, create_graph=True, retain_graph=True,
#                                grad_outputs=weight)
#     u1x0 = u1x[:,0:1]
#     u1x1 = u1x[:,1:2]
#     u1x0x, = torch.autograd.grad(u1x0, x, create_graph=True, retain_graph=True,
#                                  grad_outputs=weight)
#     u1x0x0 = u1x0x[:,0:1]
#     u1x1x, = torch.autograd.grad(u1x1, x, create_graph=True, retain_graph=True,
#                                  grad_outputs=weight)
#     u1x1x1 = u1x1x[:,1:2]

#     p = uu(x, 2)
#     px, = torch.autograd.grad(p, x, create_graph=True, retain_graph=True,
#                               grad_outputs=weight)
#     px0 = px[:,0:1]
#     px1 = px[:,1:2]

#     return -1/Re * (u0x0x0 + u0x1x1) + u0*u0x0 + u1*u0x1 + px0, \
#            -1/Re * (u1x0x0 + u1x1x1) + u0*u1x0 + u1*u1x1 + px1, \
#            u0x0 + u1x1

# ----------------------------------------------------------------------------------------------------
class InSet():
    """ data structure of training set on the interior of computational domain
    
    parameters
    bounds: spatial bounds of computational domain
    nx: number of data points in spatial dimension
    gp_num: number of gaussian points
    Re: Renold number
    device: device storing data

    attributes
    bounds: spatial bounds of computational domain
    nx: number of data points in spatial dimension
    gp_num: number of gaussian points
    Re: Renold number
    device: device storing data
    size: number of data points
    dim: spatial dimension
    x: coordinates of data points, which are located on the supports of test functions
    r: right hand side of governing equation
    v_size: number of test functions
    v_ind: indices of integration points
    v: values of test functions
    vx: spatial gradients of test functions
    weight_grad: weight for calculate gradients
    """
    def __init__(self, bounds, nx, device, from_out=None):
        self.bounds = bounds
        self.nx = nx

        if from_out is None:
            lb = bounds[:,0].cpu().numpy()
            ub = bounds[:,1].cpu().numpy()
            self.x = torch.from_numpy(lb + (ub - lb) * lhs(2, nx))
        else:
            self.x = from_out.detach().clone().cpu()

        self.x.requires_grad = True
        self.weight_grad = torch.ones_like(self.x[:,0:1])
        self.weight_grad = self.weight_grad.to(device)
        self.x = self.x.to(device)


class BdSet():
    """ data struture of training set on the domain boundary

    parameters
    bounds: spatial bounds of computational domain
    nx: number of data points in spatial dimension
    gp_num: number of gaussian points
    Re: Renold number
    device: device storing data
    
    attributes
    bounds: spatial bounds of computational domain
    nx: number of data points in spatial dimension
    gp_num: number of gaussian points
    Re: Renold number
    device: device storing data
    size: number of data points
    x: coordinates of data points
    r0, r1: right hand side of boundary equation
    """
    def __init__(self, bounds, nx, gp_num, Re, device):
        self.bounds = bounds
        self.nx = nx
        self.gp_num = gp_num
        self.Re = Re
        self.device = device

        if self.gp_num==1:
            self.gp_wei = [2.0]
            self.gp_pos = [0.5]
        if self.gp_num==2:
            self.gp_wei = [1.0, 1.0]
            self.gp_pos = [(1-0.5773502692)/2, (1+0.5773502692)/2]

        self.size = 2*(self.nx[0]+self.nx[1]) * self.gp_num
        self.dim = self.bounds.shape[0]
        self.x = torch.zeros(self.size,self.dim)
        self.hx = (self.bounds[:,1]-self.bounds[:,0])/self.nx

        self.r0 = torch.zeros(self.size,1)
        self.r1 = torch.zeros(self.size,1)

        self.weighted0 = torch.ones(self.size,1)
        self.weighted1 = torch.ones(self.size,1)

        m = 0
        for i in range(self.nx[0]):
            for ii in range(self.gp_num):
                self.x[m,0] = self.bounds[0,0] + (i+self.gp_pos[ii])*self.hx[0]
                self.x[m,1] = self.bounds[1,0]
                m += 1
        for j in range(self.nx[1]):
            for jj in range(self.gp_num):
                self.x[m,0] = self.bounds[0,1]
                self.x[m,1] = self.bounds[1,0] + (j+self.gp_pos[jj])*self.hx[1]
                m += 1
        for i in range(self.nx[0]):
            for ii in range(self.gp_num):
                self.x[m,0] = self.bounds[0,0] + (i+self.gp_pos[ii])*self.hx[0]
                self.x[m,1] = self.bounds[1,1]
                self.weighted0[m] = 2 - 4*torch.abs(self.x[m,0]-0.5)
                m += 1
        for j in range(self.nx[1]):
            for jj in range(self.gp_num):
                self.x[m,0] = self.bounds[0,0]
                self.x[m,1] = self.bounds[1,0] + (j+self.gp_pos[jj])*self.hx[1]
                m += 1

        self.r0 = uu(self.x,0,self.Re)
        self.r1 = uu(self.x,1,self.Re)

        self.x = self.x.to(self.device)
        self.r0 = self.r0.to(self.device)
        self.r1 = self.r1.to(self.device)
        self.weighted0 = self.weighted0.to(self.device)
        self.weighted1 = self.weighted1.to(self.device)

class TeSet():
    """ data structure of test points

    parameters
    bounds: spatial bounds of computational domain
    nx: number of data points in spatial dimension
    Re: Renold number
    device: device storing data
    
    attributes
    bounds: spatial bounds of computational domain
    nx: number of data points in spatial dimension
    Re: Renold number
    device: device storing data
    size: number of data points
    dim: spatial dimension
    x: coordinates of data points
    u0a, u1a: values of true solution of data points
    """
    def __init__(self, bounds, nx, Re, device):
        self.bounds = bounds
        self.nx = nx
        self.Re = Re
        self.device = device

        Ghia_Table_I_x = [[0.5,0.0000], [0.5,0.0547], [0.5,0.0625], [0.5,0.0703], [0.5,0.1016], 
                          [0.5,0.1719], [0.5,0.2813], [0.5,0.4531], [0.5,0.5000], [0.5,0.6172],
                          [0.5,0.7344], [0.5,0.8516], [0.5,0.9531], [0.5,0.9609], [0.5,0.9688],
                          [0.5,0.9766], [0.5,1.0000]]
        if Re == 100:
            Ghia_Table_I_u0a = [[0.0000], [-0.03717], [-0.04192], [-0.04775], [-0.06434], 
                                [-0.10150], [-0.15662], [-0.21090], [-0.20581], [-0.13641],
                                [0.00332], [0.23151], [0.68717], [0.73722], [0.78871],
                                [0.84123], [1.0000]]
        elif Re == 400:
            Ghia_Table_I_u0a = [[0.0000], [-0.08186], [-0.09266], [-0.10338], [-0.14612], 
                                [-0.24299], [-0.32726], [-0.17119], [-0.11477], [0.02135],
                                [0.16256], [0.29093], [0.55892], [0.61756], [0.68439],
                                [0.75837], [1.0000]]
        elif Re == 1000:
            Ghia_Table_I_u0a = [[0.0000], [-0.18109], [-0.20196], [-0.22220], [-0.29730], 
                                [-0.38289], [-0.27805], [-0.10648], [-0.06080], [0.05702],
                                [0.18719], [0.33304], [0.46604], [0.51117], [0.57492],
                                [0.65928], [1.0000]]
        
        
        
        if Re == 100:
            Ghia_Table_II_x = [[0.0000,0.5], [0.0625,0.5], [0.0703,0.5], [0.0781,0.5], [0.0938,0.5],
                                [0.1563,0.5], [0.2266,0.5], [0.2344,0.5], [0.5000,0.5], [0.8047,0.5],
                                [0.8594,0.5], [0.9063,0.5], [0.9453,0.5], [0.9531,0.5], [0.9609,0.5],
                                [0.9688,0.5], [1.0000,0.5]]
            Ghia_Table_II_u1a = [[0.00000], [0.09233], [0.10091], [0.10890], [0.12317],
                                    [0.16077], [0.17507], [0.17527], [0.05454], [-0.24533],
                                    [-0.22445], [-0.16914], [-0.10313], [-0.08864], [-0.07391],
                                    [-0.05906], [0.00000]]
        elif Re == 400:
            Ghia_Table_II_x = [[0.0000,0.5], [0.0625,0.5], [0.0703,0.5], [0.0781,0.5], [0.0938,0.5],
                                [0.1563,0.5], [0.2266,0.5], [0.2344,0.5], [0.5000,0.5], [0.8047,0.5],
                                [0.8594,0.5], [0.9453,0.5], [0.9531,0.5], [0.9609,0.5],
                                [0.9688,0.5], [1.0000,0.5]]
            Ghia_Table_II_u1a = [[0.00000], [0.18360], [0.19713], [0.20920], [0.22965],
                                    [0.28124], [0.30203], [0.30174], [0.05186], [-0.38598],
                                    [-0.44993], [-0.22847], [-0.19254], [-0.15663],
                                    [-0.12146], [0.00000]]
        elif Re == 1000:
            Ghia_Table_II_x = [[0.0000,0.5], [0.0625,0.5], [0.0703,0.5], [0.0781,0.5], [0.0938,0.5],
                                [0.1563,0.5], [0.2266,0.5], [0.2344,0.5], [0.5000,0.5], [0.8047,0.5],
                                [0.8594,0.5], [0.9063,0.5], [0.9453,0.5], [0.9531,0.5], [0.9609,0.5],
                                [0.9688,0.5], [1.0000,0.5]]
            Ghia_Table_II_u1a = [[0.00000], [0.27485], [0.29012], [0.30353], [0.32627],
                                    [0.37095], [0.33075], [0.32235], [0.02526], [-0.31966],
                                    [-0.42665], [-0.51550], [-0.39188], [-0.33714], [-0.27669],
                                    [-0.21388], [0.00000]]

        
        self.Ghia_I_x = torch.tensor(Ghia_Table_I_x)
        self.Ghia_I_u0a = torch.tensor(Ghia_Table_I_u0a)

        self.Ghia_II_x = torch.tensor(Ghia_Table_II_x)
        self.Ghia_II_u1a = torch.tensor(Ghia_Table_II_u1a)

        self.Ghia_I_x = self.Ghia_I_x.to(self.device)
        self.Ghia_I_u0a = self.Ghia_I_u0a.to(self.device)

        self.Ghia_II_x = self.Ghia_II_x.to(self.device)
        self.Ghia_II_u1a = self.Ghia_II_u1a.to(self.device)




class extended_BdSet():
    """ data struture of training set on the domain boundary

    parameters
    bounds: spatial bounds of computational domain
    nx: number of data points in spatial dimension
    gp_num: number of gaussian points
    Re: Renold number
    device: device storing data
    
    attributes
    bounds: spatial bounds of computational domain
    nx: number of data points in spatial dimension
    gp_num: number of gaussian points
    Re: Renold number
    device: device storing data
    size: number of data points
    x: coordinates of data points
    r0, r1: right hand side of boundary equation
    """
    def __init__(self, bounds, nx, gp_num, Re, device):
        self.bounds = bounds
        self.nx = nx
        self.gp_num = gp_num
        self.Re = Re
        self.device = device

        if self.gp_num==1:
            self.gp_wei = [2.0]
            self.gp_pos = [0.5]
        if self.gp_num==2:
            self.gp_wei = [1.0, 1.0]
            self.gp_pos = [(1-0.5773502692)/2, (1+0.5773502692)/2]

        self.size = self.nx[2] * (2*(self.nx[0]+self.nx[1]) * self.gp_num)
        self.dim = self.bounds.shape[0]
        self.x = torch.zeros(self.size,self.dim)
        self.hx = (self.bounds[:,1]-self.bounds[:,0])/self.nx

        self.r0 = torch.zeros(self.size,1)
        self.r1 = torch.zeros(self.size,1)

        self.weighted0 = torch.ones(self.size,1)
        self.weighted1 = torch.ones(self.size,1)

        m = 0
        for nu in range(self.nx[2]):
            for i in range(self.nx[0]):
                for ii in range(self.gp_num):
                    self.x[m,0] = self.bounds[0,0] + (i+self.gp_pos[ii])*self.hx[0]
                    self.x[m,1] = self.bounds[1,0]
                    self.x[m,2] = self.bounds[2,0] + (nu+0.5)*self.hx[2]
                    m += 1
            for j in range(self.nx[1]):
                for jj in range(self.gp_num):
                    self.x[m,0] = self.bounds[0,1]
                    self.x[m,1] = self.bounds[1,0] + (j+self.gp_pos[jj])*self.hx[1]
                    self.x[m,2] = self.bounds[2,0] + (nu+0.5)*self.hx[2]
                    m += 1
            for i in range(self.nx[0]):
                for ii in range(self.gp_num):
                    self.x[m,0] = self.bounds[0,0] + (i+self.gp_pos[ii])*self.hx[0]
                    self.x[m,1] = self.bounds[1,1]
                    self.x[m,2] = self.bounds[2,0] + (nu+0.5)*self.hx[2]
                    self.weighted0[m] = 2 - 4*torch.abs(self.x[m,0]-0.5)
                    m += 1
            for j in range(self.nx[1]):
                for jj in range(self.gp_num):
                    self.x[m,0] = self.bounds[0,0]
                    self.x[m,1] = self.bounds[1,0] + (j+self.gp_pos[jj])*self.hx[1]
                    self.x[m,2] = self.bounds[2,0] + (nu+0.5)*self.hx[2]
                    m += 1

        self.r0 = uu(self.x,0,self.Re)
        self.r1 = uu(self.x,1,self.Re)

        self.x = self.x.to(self.device)
        self.r0 = self.r0.to(self.device)
        self.r1 = self.r1.to(self.device)
        self.weighted0 = self.weighted0.to(self.device)
        self.weighted1 = self.weighted1.to(self.device)