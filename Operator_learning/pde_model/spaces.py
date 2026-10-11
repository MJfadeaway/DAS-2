from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic 


class FiniteChebyshev:
    def __init__(self, N=100, M=1):
        self.N = N
        self.M = M

    def random(self, n):
        return 2 * self.M * np.random.rand(n, self.N) - self.M

    def eval_u_one(self, a, x):
        return np.polynomial.chebyshev.chebval(2 * x - 1, a)

    def eval_u(self, a, sensors):
        return np.polynomial.chebyshev.chebval(2 * np.ravel(sensors) - 1, a.T)
