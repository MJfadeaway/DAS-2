from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np


def flatten_sum(logps):
    assert len(logps.shape) == 2
    return torch.sum(logps, [1], keepdims=True)

def log_standard_Gaussian(x):
    return flatten_sum(-0.5*(np.log(2.*np.pi)+x**2))