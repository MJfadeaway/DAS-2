from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pathos.pools import ProcessPool
from scipy import interpolate
from scipy.integrate import solve_ivp 
from icecream import ic 


class ODESystem(object):
    def __init__(self, g, s0, T):
        self.g = g
        self.s0 = s0
        self.T = T

    def gen_operator_data(self, space, m, Q, num, mode, features=None):
        print("Generating {} operator data...".format(mode), flush=True)

        if features is None:
            print('please feed features!')
            features = space.random(num)
        else:
            num = features.shape[0] 

        sensors = np.linspace(0, self.T, num=m)[:, None]
        sensor_values = space.eval_u(features, sensors)
        # x = self.T * np.random.rand(num)[:, None]

        x = np.tile(np.linspace(0+0.5*self.T/Q, self.T-0.5*self.T/Q, num=Q), (num, 1)) 

        u_ = space.eval_u(features, np.linspace(0+0.5*self.T/Q, self.T-0.5*self.T/Q, num=Q)[:, None])

        if mode == 'train':
            return {
                    'features': features,
                    'sensor_values': sensor_values,
                    'collocation_points': x,
                    'source_on_collocation_points': u_ 
                    }
        elif mode == 'test':
            y = self.eval_s_space(space, features, x) 
            return {
                    'features': features,
                    'sensor_values': sensor_values,
                    'collocation_points': x, 
                    'source_on_collocation_points': u_, 
                    'solution_on_collocation_points': y
                    }
        else:
            return 

    def eval_s_space(self, space, features, x):
        """For a list of functions in `space` represented by `features`
        and a list `x`, compute the corresponding list of outputs.
        """

        def f(feature, xi):
            return self.eval_s_func(lambda t: space.eval_u_one(feature, t), xi)

        p = ProcessPool(nodes=16)
        res = p.map(f, features, x)
        return np.array(list(res))

    def eval_s_func(self, u, x):
        """For an input function `u` and a list `x`, compute the corresponding list of outputs.
        """
        res = map(lambda xi: self.eval_s(u, xi), x)
        return np.array(list(res))

    def eval_s(self, u, tf):
        """Compute `s`(`tf`) for an input function `u`.
        """

        def f(t, y):
            return self.g(y, u(t), t)

        sol = solve_ivp(f, [0, tf], self.s0, method="RK45")
        return sol.y[0, -1]