from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from icecream import ic 
import torch.nn.functional as F
import math 


# two-hidden-layer neural network
class NN2(torch.nn.Module):
    def __init__(self, n_dim, n_width=32, n_out=None):
        super(NN2, self).__init__()
        self.n_dim = n_dim 
        self.n_width = n_width
        self.n_out = n_out

        self.l_1 = torch.nn.Linear(n_dim, n_width)
        self.l_2 = torch.nn.Linear(n_width, n_width)
        self.l_f = torch.nn.Linear(n_width, n_out)

    def forward(self, inputs):
        # relu with low regularity
        x = F.relu(self.l_1(inputs))
        x = F.relu(self.l_2(x))

        # tanh with high regularity
        #x = tf.nn.tanh(self.l_1(inputs))
        #x = tf.nn.tanh(self.l_2(x))
        #x = tf.nn.relu(self.l_3(x))
        #x = tf.nn.relu(self.l_4(x))

        x = self.l_f(x)

        return x


class affine_coupling(torch.nn.Module):
    def __init__(self, n_dim, n_split_at, n_width = 32):
        super(affine_coupling, self).__init__()
        self.n_dim = n_dim 
        self.n_split_at = n_split_at 
        self.n_width = n_width 

        self.f = NN2(n_split_at, n_width, (n_dim-n_split_at)*2)
        self.log_gamma = torch.nn.Parameter(torch.zeros(1, n_dim-n_split_at))

    def forward(self, inputs, logdet=None, reverse=False):
        z = inputs
        n_split_at = self.n_split_at

        alpha = 0.6

        if not reverse:
            z1 = z[:,:n_split_at]
            z2 = z[:,n_split_at:]

            h = self.f(z1)
            shift = h[:,0::2]

            scale = alpha*torch.tanh(h[:,1::2])
            #shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
            shift = torch.exp(torch.clamp(self.log_gamma, -5.0, 5.0))*torch.tanh(shift)
            z2 = z2 + scale * z2 + shift
            if logdet is not None:
                dlogdet = torch.sum(torch.log(scale + torch.ones_like(scale)),
                                        [1], keepdims=True)
                
            z = torch.cat((z1,z2), 1)
            
        else:
            z1 = z[:,:n_split_at]
            z2 = z[:,n_split_at:]

            h = self.f(z1)
            shift = h[:,0::2]

            # resnet-like trick
            # we suppressed both the scale and the shift.
            scale = alpha*torch.tanh(h[:,1::2])
            #shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
            shift = torch.exp(torch.clamp(self.log_gamma, -5.0, 5.0))*torch.tanh(shift)
            z2 = (z2 - shift) / (torch.ones_like(scale) + scale)
            if logdet is not None:
                dlogdet = - torch.sum(torch.log(scale + torch.ones_like(scale)),
                                        [1], keepdims=True)

            z = torch.cat((z1,z2), 1)

        if logdet is not None:
           return z, logdet + dlogdet
        
        return z 


class actnorm(torch.nn.Module):
    def __init__(self, n_dim, scale = 1.0, logscale_factor=3.0):
        super(actnorm, self).__init__()
        self.n_dim = n_dim 
        self.scale = scale
        self.logscale_factor = logscale_factor

        self.data_init = False 

        self.b = torch.nn.Parameter(torch.zeros(1, n_dim))
        self.logs = torch.nn.Parameter(torch.zeros(1, n_dim))

    def forward(self, inputs, logdet=None, reverse=False):
        assert inputs.shape[-1] == self.n_dim 

        if not self.data_init:
            x_mean = torch.mean(inputs, [0], keepdims=True)
            x_var = torch.mean(torch.square(inputs-x_mean), [0], keepdims=True)

            self.b_init = - x_mean 
            self.logs_init = torch.log(self.scale/(torch.sqrt(x_var)+1e-6))/self.logscale_factor

            self.b_init = self.b_init.detach()
            self.logs_init = self.logs_init.detach() 

            self.data_init = True 

        if not reverse:
            x = inputs + (self.b + self.b_init)
            #x = x * tf.exp(self.logs + self.logs_init)
            x = x * torch.exp(torch.clamp(self.logs + self.logs_init, -5., 5.))
        else:
            #x = inputs * tf.exp(-self.logs - self.logs_init)
            x = inputs * torch.exp(-torch.clamp(self.logs + self.logs_init, -5., 5.))
            x = x - (self.b + self.b_init)

        if logdet is not None:
            #dlogdet = tf.reduce_sum(self.logs + self.logs_init)
            dlogdet = torch.sum(torch.clamp(self.logs + self.logs_init, -5., 5.))
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x
    
    def reset_data_initialization(self):
        self.data_init = False 


class flow_mapping(torch.nn.Module):
    def __init__(self, n_dim, n_depth, n_split_at, n_width=32, n_bins=16, **kwargs):
        super(flow_mapping, self).__init__()
        self.n_dim = n_dim 
        self.n_depth = n_depth 
        self.n_split_at = n_split_at 
        self.n_width = n_width 
        self.n_bins = n_bins 

        assert n_depth % 2 == 0

        self.scale_layers = torch.nn.ModuleList()
        self.affine_layers = torch.nn.ModuleList() 

        sign = -1
        for i in range(self.n_depth):
            self.scale_layers.append(actnorm(n_dim)) 
            sign *= -1
            i_split_at = (self.n_split_at*sign + self.n_dim) % self.n_dim
            self.affine_layers.append(affine_coupling(n_dim, 
                                                     i_split_at,
                                                     n_width=self.n_width))
    
    def forward(self, inputs, logdet=None, reverse=False):
        assert inputs.shape[-1] == self.n_dim 

        if not reverse:
            z = inputs 
            for i in range(self.n_depth):
                z = self.scale_layers[i](z, logdet)
                if logdet is not None:
                    z, logdet = z 

                z = self.affine_layers[i](z, logdet)
                if logdet is not None:
                    z, logdet = z 

                z = torch.flip(z, dims=(1,))

        else:
            z = inputs 

            for i in reversed(range(self.n_depth)):
                z = torch.flip(z, dims=(1,))

                z = self.affine_layers[i](z, logdet, reverse=True)
                if logdet is not None:
                    z, logdet = z 

                z = self.scale_layers[i](z, logdet, reverse=True)
                if logdet is not None:
                    z, logdet = z 

        if logdet is not None:
            return z, logdet 
        return z 
    
    def actnorm_data_initialization(self):
        for i in range(self.n_depth):
            self.scale_layers[i].reset_data_initialization()


class Bounded_support_mapping(torch.nn.Module):
    def __init__(self, n_dim, lb, hb):
        super(Bounded_support_mapping, self).__init__()
        self.n_dim = n_dim 
        self.lb = lb
        self.hb = hb

        self.logistic_layer = Logistic_mapping(self.n_dim)
        self.affine_linear_layer = Affine_linear_mapping(self.n_dim, self.lb, self.hb)

    def forward(self, inputs, logdet=None, reverse=False):
        assert inputs.shape[-1] == self.n_dim 
        x = inputs

        if not reverse:
            x = self.affine_linear_layer(x, logdet)
            if logdet is not None:
                x, logdet = x
            x = self.logistic_layer(x, logdet)
            if logdet is not None:
                x, logdet = x

        else:
            x = self.logistic_layer(x, logdet, reverse=True)
            if logdet is not None:
                x, logdet = x
            x = self.affine_linear_layer(x, logdet, reverse=True)
            if logdet is not None:
                x, logdet = x

        if logdet is not None:
            return x, logdet

        return x
    
class Affine_linear_mapping(torch.nn.Module):
    def __init__(self, n_dim, lb, hb):
        super(Affine_linear_mapping, self).__init__()
        self.n_dim = n_dim 
        self.lb = lb
        self.hb = hb

    def forward(self, inputs, logdet=None, reverse=False):
        assert self.n_dim == inputs.shape[-1]
        x = inputs
        # mapping from [lb, hb] to [0,1]^d for PDE: y = (x-l) / (h - l)
        if not reverse:
            x = x / (self.hb - self.lb)
            x = x - self.lb / (self.hb - self.lb)
        else:
            x = x + self.lb / (self.hb - self.lb)
            x = x * (self.hb - self.lb)

        if logdet is not None:
            dlogdet = self.n_dim * np.log(1.0/(self.hb-self.lb)) * torch.ones_like(x[:,0:1])
            if reverse:
                dlogdet *= -1.0
            return x, logdet + dlogdet

        return x
    

class Logistic_mapping(torch.nn.Module):
    """
    Logistic mapping, (-inf, inf) --> (0, 1):
    y = (tanh(x/2) + 1) / 2 = e^x/(1 + e^x)
    derivate: dy/dx = y* (1-y)
    inverse: x = log(y) - log(1-y)

    For PDE, data to prior direction: [a,b] ---> (-inf, inf)
    So we need to use an affine linear mapping first and then use logistic mapping
    """
    def __init__(self, n_dim):
        super(Logistic_mapping, self).__init__()
        self.n_dim = n_dim 
        self.s_init = 2.0

    # the direction of this mapping is not related to the flow
    # direction between the data and the prior
    def forward(self, inputs, logdet=None, reverse=False):
        assert inputs.shape[-1] == self.n_dim 
        x = inputs

        if not reverse:
            if self.n_dim == 2:
                x = torch.clamp(x, 1.0e-10, 1.0-1.0e-10)
            else:
                x = torch.clamp(x, 1.0e-6, 1.0-1.0e-6)
            tp1 = torch.log(x)
            tp2 = torch.log(1 - x)
            #x = (self.s_init + self.s) / 2.0 * (tp1 - tp2)
            x = tp1 - tp2
            if logdet is not None:
                #tp =  tf.math.log((self.s+self.s_init)/2.0) - tp1 - tp2
                tp =  - tp1 - tp2
                dlogdet = torch.sum(tp, axis=[1], keepdims=True)
                return x, logdet + dlogdet

        else:
            x = (torch.tanh(x / self.s_init) + 1.0) / 2.0

            if logdet is not None:
                if self.n_dim == 2:
                    x = torch.clamp(x, 1.0e-10, 1.0-1.0e-10)
                else:
                    x = torch.clamp(x, 1.0e-6, 1.0-1.0e-6)
                #tp = tf.math.log(x) + tf.math.log(1-x) + tf.math.log(2.0/(self.s+self.s_init))
                tp = torch.log(x) + torch.log(1-x)
                dlogdet = torch.sum(tp, axis=[1], keepdims=True)
                return x, logdet + dlogdet
 
        return x
    

# squeezing layer - KR rearrangement
class squeezing(torch.nn.Module):
    def __init__(self, n_dim, n_cut=1):
        super(squeezing, self).__init__()
        self.n_dim = n_dim
        self.n_cut = n_cut
        self.x = None

    def forward(self, inputs, reverse=False):
        z = inputs
        n_length = z.shape[-1]

        if not reverse:
            if n_length < self.n_cut:
                raise Exception()

            if self.n_dim == n_length:
                if self.n_dim > self.n_cut:
                    if self.x is not None:
                        raise Exception()
                    else:
                        self.x = z[:, (n_length - self.n_cut):]
                        z = z[:, :(n_length - self.n_cut)]
                else:
                    self.x = None
            elif n_length <= self.n_cut:
                z = torch.cat((z, self.x), 1)
                self.x = None
            else:
                cut = z[:, (n_length - self.n_cut):]
                self.x = torch.cat((cut, self.x), 1)
                z = z[:, :(n_length - self.n_cut)]
        else:
            if self.n_dim == n_length:
                n_start = self.n_dim % self.n_cut
                if n_start == 0:
                    n_start += self.n_cut
                self.x = z[:, n_start:]
                z = z[:, :n_start]

            else:
                x_length = self.x.shape[-1]
                if x_length < self.n_cut:
                    raise Exception()

                cut = self.x[:, :self.n_cut]
                z = torch.cat((z, cut), 1)

                if (x_length - self.n_cut) == 0:
                    self.x = None
                else:
                    self.x = self.x[:, self.n_cut:]
        return z
    

class scale_and_CDF(torch.nn.Module):
  def __init__(self, n_dim, n_bins=16):
    super(scale_and_CDF, self).__init__()
    self.n_dim = n_dim 
    self.n_bins = n_bins

    self.scale_layer = actnorm(n_dim)
    self.cdf_layer = CDF_quadratic(self.n_dim, self.n_bins)

  def forward(self, inputs, logdet=None, reverse=False):
    z = inputs
    assert z.shape[-1] == self.n_dim 
    if not reverse:
      z = self.scale_layer(z, logdet)
      if logdet is not None:
        z, logdet = z

      z = self.cdf_layer(z, logdet)
      if logdet is not None:
        z, logdet = z
    else:
      z = self.cdf_layer(z, logdet, reverse=True)
      if logdet is not None:
        z, logdet = z

      z = self.scale_layer(z, logdet, reverse=True)
      if logdet is not None:
        z, logdet = z

    if logdet is not None:
      return z, logdet

    return z

  def actnorm_data_initialization(self):
    self.scale_layer.reset_data_initialization()





# mapping defined by a piecewise quadratic cumulative distribution function (CDF)
# Assume that each dimension has a compact support [0,1]
# CDF(x) maps [0,1] to [0,1], where the prior uniform distribution is defined.
# Since x is defined on (-inf,+inf), we only consider a CDF() mapping from
# the interval [-bound, bound] to [-bound, bound], and leave alone other points.
# The reason we do not consider a mapping from (-inf,inf) to (0,1) is the
# singularity induced by the mapping.
class CDF_quadratic(torch.nn.Module):
    def __init__(self, n_dim, n_bins, r=1.2, bound=50.0):
        super(CDF_quadratic, self).__init__()

        assert n_bins % 2 == 0

        self.n_dim = n_dim 
        self.n_bins = n_bins

        # generate a nonuniform mesh symmetric to zero,
        # and increasing by ratio r away from zero.
        self.bound = bound
        self.r = r

        m = n_bins/2
        x1L = bound*(r-1.0)/(r**m-1.0)

        index = torch.reshape(torch.arange(0, self.n_bins+1, dtype=torch.float32),(-1,1))
        index -= m
        xr = torch.where(index>=0, (1.-torch.pow(r, index))/(1.-r),
                      (1.-torch.pow(r,torch.abs(index)))/(1.-r))
        xr = torch.where(index>=0, x1L*xr, -x1L*xr)
        xr = torch.reshape(xr,(-1,1))
        xr = (xr + bound)/2.0/bound

        self.x1L = x1L/2.0/bound
        self.mesh = torch.cat([torch.tensor([[0.0]]), torch.reshape(xr[1:-1,0],(-1,1)), torch.tensor([[1.0]])],0) 
        self.elmt_size = torch.reshape(self.mesh[1:] - self.mesh[:-1],(-1,1))

        self.n_length = n_dim 
        self.p = torch.nn.Parameter(torch.zeros(self.n_bins-1, self.n_length))

    def forward(self, inputs, logdet=None, reverse=False):

        assert inputs.shape[-1] == self.n_dim 

        # normalize the PDF
        self._pdf_normalize()

        x = inputs
        if not reverse:
            # rescale such points in [-bound, bound] will be mapped to [0,1]
            x = (x + self.bound) / 2.0 / self.bound

            # cdf mapping
            x = self._cdf(x, logdet)
            if logdet is not None:
                x, logdet = x

            # maps [0,1] back to [-bound, bound]
            x = x * 2.0 * self.bound - self.bound
        else:
            # rescale such points in [-bound, bound] will be mapped to [0,1]
            x = (x + self.bound) / 2.0 / self.bound

            # cdf mapping
            x = self._cdf_inv(x, logdet)
            if logdet is not None:
                x, logdet = x

            # maps [0,1] back to [-bound, bound]
            x = x * 2.0 * self.bound - self.bound
        if logdet is not None:
            return x, logdet

        return x

    # normalize the piecewise representation of pdf
    def _pdf_normalize(self):
        # peicewise pdf
        p0 = torch.ones(1, self.n_length, device=self.p.device)
        self.pdf = p0
        self.mesh = self.mesh.to(self.p.device)
        self.elmt_size = self.elmt_size.to(self.p.device) 

        px = torch.exp(self.p)*(self.elmt_size[:-1]+self.elmt_size[1:])/2.0
        px = (1 - self.elmt_size[0])/torch.sum(px, 0, keepdims=True)
        px = px*torch.exp(self.p)
        self.pdf = torch.cat([self.pdf, px], 0)
        self.pdf = torch.cat([self.pdf, p0], 0)

        # probability in each element
        cell = (self.pdf[:-1,:] + self.pdf[1:,:])/2.0*self.elmt_size
        # CDF - contribution from previous elements.
        r_zeros= torch.zeros(1, self.n_length, device=self.p.device) 
        self.F_ref = r_zeros
        for i in range(1, self.n_bins):
            tp  = torch.sum(cell[:i,:], 0, keepdims=True)
            self.F_ref = torch.cat([self.F_ref, tp], 0)

    # the cdf is a piecewise quadratic function.
    def _cdf(self, x, logdet=None):
        x_sign = torch.sign(x-0.5)
        m = torch.floor(torch.log(torch.abs(x-0.5)*(self.r-1)/self.x1L + 1.0)/np.log(self.r))
        k_ind = torch.where(x_sign >= 0, self.n_bins/2 + m, self.n_bins/2 - m - 1)
        k_ind = k_ind.to(torch.int32)

        cover = torch.where(k_ind*(k_ind-self.n_bins+1)<=0, 1.0, 0.0)

        k_ind = torch.where(k_ind < 0, 0, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins-1), self.n_bins-1, k_ind)

        v1 = torch.reshape(self.pdf[:,0][k_ind[:,0]],(-1,1))
        for i in range(1, self.n_length):
            tp = torch.reshape(self.pdf[:,i][k_ind[:,i]],(-1,1))
            v1 = torch.cat([v1, tp], 1)

        v2 = torch.reshape(self.pdf[:,0][k_ind[:,0]+1],(-1,1))
        for i in range(1, self.n_length):
            tp = torch.reshape(self.pdf[:,i][k_ind[:,i]+1],(-1,1))
            v2 = torch.cat([v2, tp], 1)

        xmodi = torch.reshape(x[:,0] - self.mesh[:,0][k_ind[:, 0]], (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(x[:,i] - self.mesh[:,0][k_ind[:, i]], (-1, 1))
            xmodi = torch.cat([xmodi, tp], 1)

        h_list = torch.reshape(self.elmt_size[:,0][k_ind[:,0]],(-1,1))
        for i in range(1, self.n_length):
            tp = torch.reshape(self.elmt_size[:,0][k_ind[:,i]],(-1,1))
            h_list = torch.cat([h_list, tp], 1)

        F_pre = torch.reshape(self.F_ref[:, 0][k_ind[:, 0]], (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(self.F_ref[:, i][k_ind[:, i]], (-1, 1))
            F_pre = torch.cat([F_pre, tp], 1)

        y = torch.where(cover>0, F_pre + xmodi**2/2.0*(v2-v1)/h_list + xmodi*v1, x)
       
        if logdet is not None:
            dlogdet = torch.where(cover > 0, xmodi * (v2 - v1) / h_list + v1, 1.0)
            dlogdet = torch.sum(torch.log(dlogdet), axis=[1], keepdims=True)
            return y, logdet + dlogdet

        return y

    # inverse of the cdf
    def _cdf_inv(self, y, logdet=None):
        xr = torch.broadcast_to(self.mesh, [self.n_bins+1, self.n_length])
        yr1 = self._cdf(xr)

        p0 = torch.zeros(1, self.n_length, device=self.p.device)
        p1 = torch.ones(1, self.n_length, device=self.p.device)
        yr = torch.cat([p0, yr1[1:-1,:], p1], 0)

        k_ind = torch.searchsorted(yr.T, y.T, side='right')
        k_ind = k_ind.T 
        k_ind = k_ind.to(torch.int32)
        k_ind -= 1

        cover = torch.where(k_ind*(k_ind-self.n_bins+1)<=0, 1.0, 0.0)

        k_ind = torch.where(k_ind < 0, 0, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins-1), self.n_bins-1, k_ind)

        c_cover = torch.reshape(cover[:,0], (-1,1))
        v1 = torch.where(c_cover > 0, torch.reshape(self.pdf[:,0][k_ind[:,0]],(-1,1)), -1.0)
        for i in range(1, self.n_length):
            c_cover = torch.reshape(cover[:,i], (-1,1))
            tp = torch.where(c_cover > 0, torch.reshape(self.pdf[:,i][k_ind[:,i]],(-1,1)), -1.0)
            v1 = torch.cat([v1, tp], 1)

        c_cover = torch.reshape(cover[:,0], (-1,1))
        v2 = torch.where(c_cover > 0, torch.reshape(self.pdf[:,0][k_ind[:,0]+1],(-1,1)), -2.0)
        for i in range(1, self.n_length):
            c_cover = torch.reshape(cover[:,i], (-1,1))
            tp = torch.where(c_cover > 0, torch.reshape(self.pdf[:,i][k_ind[:,i]+1],(-1,1)), -2.0)
            v2 = torch.cat([v2, tp], 1)

        ys = torch.reshape(y[:, 0] - yr[:, 0][k_ind[:, 0]], (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(y[:, i] - yr[:, i][k_ind[:, i]], (-1, 1))
            ys = torch.cat([ys, tp], 1)

        xs = torch.reshape(xr[:, 0][k_ind[:, 0]], (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(xr[:, i][k_ind[:, i]], (-1, 1))
            xs = torch.cat([xs, tp], 1)

        h_list = torch.reshape(self.elmt_size[:,0][k_ind[:,0]],(-1,1))
        for i in range(1, self.n_length):
            tp = torch.reshape(self.elmt_size[:,0][k_ind[:,i]],(-1,1))
            h_list = torch.cat([h_list, tp], 1)

        tp = 2.0*ys*h_list*(v2-v1)
        tp += v1*v1*h_list*h_list
        tp = torch.sqrt(tp) - v1*h_list
        tp = torch.where(torch.abs(v1-v2)<1.0e-6, ys/v1, tp/(v2-v1))
        tp += xs

        x = torch.where(cover > 0, tp, y)

        if logdet is not None:
            tp = 2.0 * ys * h_list * (v2 - v1)
            tp += v1 * v1 * h_list * h_list
            tp = h_list/torch.sqrt(tp)

            dlogdet = torch.where(cover > 0, tp, 1.0)
            dlogdet = torch.sum(torch.log(dlogdet), axis=[1], keepdims=True)
            return x, logdet + dlogdet

        return x
