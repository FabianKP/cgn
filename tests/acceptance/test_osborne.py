"""
Tests ggn_new.solvers.GeneralizedGaussNewton on the Osborne17 problem (see Mori et al. )
"""

from math import exp
import numpy as np

import cgn
from tests.acceptance.do_test import do_test
from tests.acceptance.problem import TestProblem


m = 65
n = 11

t = np.zeros(m)
for i in range(m):
    t[i] = float(i/10)
y = np.array( [
    1.366, 1.191, 1.112, 1.013, 0.991, 0.885, 0.831, 0.847, 0.786, 0.725,
    0.746, 0.679, 0.608, 0.655, 0.616, 0.606, 0.602, 0.626, 0.651, 0.724,
    0.649, 0.649, 0.694, 0.644, 0.624, 0.661, 0.612, 0.558, 0.533, 0.495,
    0.500, 0.423, 0.395, 0.375, 0.372, 0.391, 0.396, 0.405, 0.428, 0.429,
    0.523, 0.562, 0.607, 0.653, 0.672, 0.708, 0.633, 0.668, 0.645, 0.632,
    0.591, 0.559, 0.597, 0.625, 0.739, 0.710, 0.729, 0.720, 0.636, 0.581,
    0.428, 0.292, 0.162, 0.098, 0.054
] )

C = np.zeros((2, n))
C[0, 0] = 1.0
C[0, 1] = 2.0
C[0, 3] = 3.0
C[0, 4] = 4.0
C[1, 0] = 1.0
C[1, 2] = 1.0
d = np.array([6.270063, 1.741584])

xstart = np.array([1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 4.5])
ropt = 4.01377e-2



def g(x):
    y = np.ones(m)
    for i in range(m):
            y[i] = x[0]*exp(-x[4]*t[i]) + x[1]*exp(-x[5]*((t[i]-x[8])**2)) \
                + x[2]*exp(-x[6]*((t[i]-x[9])**2)) + x[3]*exp(-x[7]*((t[i]-x[10])**2))
    return y


def gjac(x):
    jac = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            jac[i, 0] = exp(-x[4] * t[i])
            jac[i, 1] = exp(-x[5] * (t[i] - x[8]) ** 2)
            jac[i, 2] = exp(-x[7] * (t[i] - x[9]) ** 2)
            jac[i, 3] = exp(-x[7] * (t[i] - x[10]) ** 2)
            jac[i, 4] = - t[i] * x[0] * exp(-x[4] * t[i])
            jac[i, 5] = - (t[i] - x[8]) ** 2 * x[1] * exp(-x[5] * (t[i] - x[8]) ** 2)
            jac[i, 6] = - (t[i] - x[9]) ** 2 * x[2] * exp(-x[6] * (t[i] - x[9]) ** 2)
            jac[i, 7] = - (t[i] - x[10]) ** 2 * x[3] * exp(-x[7] * (t[i] - x[10]) ** 2)
            jac[i, 8] = 2.0 * x[5] * (t[i] - x[8]) * x[1] * exp(-x[5] * (t[i] - x[8]) ** 2)
            jac[i, 9] = 2.0 * x[6] * (t[i] - x[9]) * x[2] * exp(-x[6] * (t[i] - x[9]) ** 2)
            jac[i, 10] = 2.0 ** x[7] * (t[i] - x[10]) * x[3] * exp(-x[7] * (t[i] - x[10]) ** 2)
    return jac


def misfit(x):
    return g(x) - y

def misfit_jac(x):
    return gjac(x)


class OsborneProblem(TestProblem):
    def __init__(self):
        TestProblem.__init__(self)
        self._minimum = ropt
        self._tol = 1e-8
        self._problem = cgn.Problem(dims=[n], fun=misfit, jac=misfit_jac)
        # only very mild regularization
        self._problem.set_regularization(paramno=0, beta=1e-5)
        # add inequality constraints
        self._problem.add_inequality_constraint(c=C, d=d)
        self._start = [xstart]


def test_osborne():
    op = OsborneProblem()
    do_test(op)





