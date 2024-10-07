from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

import BR_lib.BR_layers as BR_layers
import BR_lib.BR_data as BR_data 


# invertible mapping based on real NVP and KR rearrangement and CDF inverse
class IM_rNVP_KR_CDF(torch.nn.Module):
    def __init__(self, n_dim, lb, hb, n_step, n_depth,
                 n_width=32,
                 shrink_rate=1.0,
                 flow_coupling=0,
                 n_bins=16,
                 rotation=False,
                 bounded_supp=False):
        super(IM_rNVP_KR_CDF, self).__init__()

        # two affine coupling layers are needed for each update of the vector
        assert n_depth % 2 == 0

        self.n_dim = n_dim # dimension of the data
        self.n_step = n_step # step size for dimension reduction
        self.n_depth = n_depth # depth for flow_mapping
        self.n_width = n_width
        self.n_bins = n_bins
        self.shrink_rate = shrink_rate
        self.flow_coupling = flow_coupling
        self.rotation = rotation
        self.bounded_supp = bounded_supp
        self.lb = lb
        self.hb = hb

        # the number of filtering stages
        self.n_stage = n_dim // n_step 
        if n_dim % n_step > 0:
            self.n_stage += 1

        n_length = n_dim 

        # flow mapping with n_stage
        self.flow_mappings = torch.nn.ModuleList() 
        for i in range(self.n_stage):
                if i == (self.n_stage-1):
                    self.flow_mappings.append(BR_layers.scale_and_CDF(n_dim=n_length, n_bins=n_bins))
                else: 
                    # flow_mapping given by such as real NVP
                    n_split_at = n_dim - (i+1) * n_step
                    self.flow_mappings.append(BR_layers.flow_mapping(n_length, 
                                              n_depth,
                                              n_split_at,
                                              n_width=n_width, 
                                              n_bins=n_bins))
                    n_width = int(n_width*self.shrink_rate)
                    n_length = n_length - n_step 

        # data will pass the squeezing layer at the end of each stage
        self.squeezing_layer = BR_layers.squeezing(n_dim, n_step)

        # data will pass the bounded support mapping first (for general PDE not FP equation)
        self.bounded_support_layer = BR_layers.Bounded_support_mapping(n_dim, lb, hb)

        # the prior distribution is the Gaussian distribution
        self.log_prior = BR_data.log_standard_Gaussian
        #self.log_prior = BR_data.log_uniform

    # computing the logarithm of the estimated pdf on the input data.
    def forward(self, inputs):
        objective = torch.zeros_like(inputs)[:,0]
        objective = torch.reshape(objective, [-1,1])

        # f(y) and log of jacobian
        z, objective = self.mapping_to_prior(inputs, objective)

        # logrithm of estimated pdf
        objective += self.log_prior(z)

        return objective

    # mapping from data to prior
    def mapping_to_prior(self, inputs, logdet=None):
        z = inputs

        # data preprocessing using bounded support layer
        z = self.bounded_support_layer(z, logdet)
        if logdet is not None:
            z, logdet = z

        for i in range(self.n_stage):
            if logdet is not None:
                z, logdet = self.flow_mappings[i](z, logdet)
            else:
                z = self.flow_mappings[i](z)
            z = self.squeezing_layer(z)

        if logdet is not None:
            return z, logdet
        else:
            return z

    # mapping from prior to data
    def mapping_from_prior(self, inputs):
        z = inputs
        for i in reversed(range(self.n_stage)):
            z = self.squeezing_layer(z, reverse=True)
            z = self.flow_mappings[i](z, reverse=True)

        # generate samples in domain [lb, hb]^d
        z = self.bounded_support_layer(z, reverse=True)
        return z

    # data initialization for actnorm layers
    def actnorm_data_initialization(self):
        for i in range(self.n_stage):
            self.flow_mappings[i].actnorm_data_initialization()

    # return samples from prior
    def draw_samples_from_prior(self, n_samples, n_dim):
        return torch.randn(n_samples, n_dim)
        #return tf.random.uniform([n_samples,n_dim], minval=-1, maxval=1, dtype=tf.float32)

