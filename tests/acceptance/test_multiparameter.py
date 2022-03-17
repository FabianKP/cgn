
from math import exp, sqrt
import numpy as np
import scipy.optimize as sciopt

import cgn
from do_test import do_test
from problem import TestProblem


def F(x, y):
    out = np.array([x[0] + exp(-x[1] + sqrt(y[0])),
                    x[0] ** 2 + 2 * x[1] + 1 - sqrt(y[0])])
    return out

def DF(x, y):
    jac = np.array([[1., -exp(-x[1]), 0.5 / sqrt(y[0])],
                    [2 * x[0], 2., - 0.5 / sqrt(y[0])]])
    return jac


def g(x):
    out = x[0] + x[0] ** 3 + x[1] + x[1] ** 2
    return np.array([out])

def Dg(x):
    jac = np.array([1 + 3 * x[0] ** 2, 1. + 2 * x[1]]).reshape((1, 2))
    return jac


class MultiParameterProblem(TestProblem):
    def __init__(self):
        TestProblem.__init__(self)
        self._tol = 1e-6
        x = cgn.Parameter(start=np.zeros(2), name="x")
        x.regop = np.array([[1., 2.], [3., 4.]])
        x.mean = np.array([1., 1.])
        x.beta = 0.1
        y = cgn.Parameter(start=np.ones(1), name="y")
        y.lb = np.array([0.1])
        # add inequality constraints
        incon = cgn.NonlinearConstraint(parameters=[x], fun=g, jac=Dg, ctype="ineq")
        self._problem = cgn.Problem(parameters=[x, y], fun=F, jac=DF, constraints=[incon])
        # Solve the problem with SLSQP for comparison
        def loss_fun(z):
            return self._problem.costfun(z[0:2], np.ones((1, )) * z[2])
        def loss_grad(z):
            return self._problem.costgrad(z[0:2], z[2] * np.ones((1, )))
        def incon_fun(z):
            c1 = g(z[0:2])
            c2 = z[2] * np.ones((1, ))
            return np.concatenate([c1, c2])
        def incon_jac(z):
            j_x = Dg(z[0:2])
            j = np.array([[j_x[0, 0], j_x[0, 1], 0.], [0., 0., 1.]])
            return j
        incon = {"type": "ineq", "fun": incon_fun, "jac": incon_jac}
        z_start = np.concatenate([x.start, y.start])
        z_ref = sciopt.minimize(fun=loss_fun, jac=loss_grad, x0=z_start, constraints=(incon)).x
        print(z_ref)
        self._minimum = loss_fun(z_ref)


def test_multiparameter():
    op = MultiParameterProblem()
    do_test(op)