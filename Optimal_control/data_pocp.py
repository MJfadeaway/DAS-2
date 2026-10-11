# ----------------------------------------------------------------------------------------------------
# Generate Data
# ----------------------------------------------------------------------------------------------------

import torch
import numpy as np
from Sampling import Geo

# ---------------------------------------------------------------------------------------------------
alpha = 0.001
mu1_min = 0.05
mu1_max = 0.45
mu2_min = 0.5
mu2_max = 2.5
ua = 0
ub = 10

def P(x):
    return torch.min(torch.max(x,ua+0*x),ub+0*x)

# y,p input: (x,y)
def yd(x):
    tmp = 0*x[:,0:1]
    mu2 = x[:,3:4]
    omega1 = (x[:,0:1]<=1)
    omega2 = (x[:,0:1]>1)
    tmp[omega1] = 1.    
    tmp[omega2] = mu2[omega2]
    return tmp

def region(x):
    x1 = x[:,0:1]
    x2 = x[:,1:2]
    mu1 = x[:,2:3]
    mu2 = x[:,3:4]
    r2 = (x1-1.5)**2 + (x2-0.5)**2
    s = (x1>=0)&(x1<=2)&(x2>=0)&(x2<=1)&(r2>=mu1**2)\
        &(mu1>=mu1_min)&(mu1<=mu1_max)&(mu2>=mu2_min)&(mu2<=mu2_max)
    return  s

bounds = np.array([-0.1,2.1, -0.1,1.1, \
                   mu1_min-0.1,mu1_max+0.1, mu2_min-0.1,mu2_max+0.1]).reshape(4,2)
Omega = Geo(region, bounds)

# ----------------------------------------------------------------------------------------------------
# Interior Set  Circle
class InSet():
    def __init__(self, size, device, sampling):
        self.device = device
        self.size = size
        
        self.area = 2*(mu2_max-mu2_min) - \
            np.pi/3*(mu1_max**2 + mu1_min**2 + mu1_max*mu1_min)
        
        self.dim = 4
      
        if sampling=='random':
            self.x = torch.from_numpy(Omega.samples(self.size)).to(self.device) 
        elif sampling=='quasi-random':
            self.x = torch.from_numpy(Omega.quasi_samples(self.size)).to(self.device) 
                  
        self.y_d = yd(self.x)     
        self.x.requires_grad = True    
        self.weight_grad = torch.ones(self.size,1)    
        self.x = self.x.to(self.device)
        self.y_d = self.y_d.to(self.device)
        self.weight_grad = self.weight_grad.to(self.device)
    
# Test set
class Te_InSet():
    def __init__(self, size, device):
        self.device = device
        self.size = size
        
        self.area = 2*(mu2_max-mu2_min) - \
            np.pi/3*(mu1_max**2 + mu1_min**2 + mu1_max*mu1_min)
            
        self.dim = 4
                        
        self.x = torch.from_numpy(Omega.samples(self.size)).to(self.device) 
     
        self.y_d = yd(self.x)      
        self.x.requires_grad = True      
        self.weight_grad = torch.ones(self.size,1)      
        self.x = self.x.to(self.device)
        self.y_d = self.y_d.to(self.device)
        self.weight_grad = self.weight_grad.to(self.device)

