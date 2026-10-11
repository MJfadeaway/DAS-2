# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 11:37:50 2021

@author: Pengfei
"""

import numpy as np
from scipy.stats import qmc # for quasi random sampling


class Geo():
    def __init__(self,region,bounds):
        self.region = region
        self.bounds = bounds
        self.dim = self.bounds.shape[0]
        
    def samples(self,N):
        x = np.zeros((N,self.dim))
        m=0
        while (m<N):
            pt = np.random.uniform(0,1,self.dim).reshape(1,-1)
            pt = pt*(self.bounds[:,1]-self.bounds[:,0])+self.bounds[:,0]
            if self.region(pt).all():
                x[m,:] = pt.ravel()
                m += 1
        return x          

    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=4*N)
        sample = sample*(self.bounds[:,1]-self.bounds[:,0])+self.bounds[:,0]
        sample = sample[self.region(sample).flatten(),:][:N,:]
        return sample   