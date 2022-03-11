
from math import exp
import numpy as np
import scipy.optimize as sciopt

import cgn
from tests.acceptance.do_test import do_test
from tests.acceptance.problem import TestProblem


n = 2


def misfit1(x: np.ndarray):
    y = np.zeros(2)
    y[0] = x[0] + exp(-x[1])
    y[1] = x[0]**2 + 2*x[1] + 1
    return y


def misfitjac1(x):
    J = np.zeros((2,2))
    J[0,0] = 1
    J[0,1] = - exp(-x[1])
    J[1,0] = 2*x[0]
    J[1,1] = 2
    return J


def constraint1(x):
    c = x[0] + x[0]**3 + x[1] + x[1]**2
    return np.array([c])


def constraintjac1(x):
    J = np.zeros((1,2))
    J[0,0] = 1 + 3*x[0]**2
    J[0,1] = 1 + 2*x[1]
    return J


class NonlinearEqualityConstraint1(TestProblem):
    def __init__(self):
        TestProblem.__init__(self)
        self._tol = 1e-6
        x = cgn.Parameter(dim=n, name="x")
        # add inequality constraints
        incon = cgn.NonlinearConstraint(parameters=[x], fun=constraint1, jac=constraintjac1, ctype="eq")
        self._problem = cgn.Problem(parameters=[x], fun=misfit1, jac=misfitjac1, constraints=[incon])
        xstart = np.zeros(2)
        self._start = [xstart]
        # Solve the problem with SLSQP for comparison
        loss_fun = self._problem.costfun
        loss_grad = self._problem.costgrad
        eqcon = {"type": "eq", "fun": constraint1, "jac": constraintjac1}
        x_ref = sciopt.minimize(fun=loss_fun, jac=loss_grad, x0=xstart, constraints=(eqcon)).x
        print(x_ref)
        self._minimum = loss_fun(x_ref)


def test_nonlinear_equality_constraint1():
    op = NonlinearEqualityConstraint1()
    do_test(op)
