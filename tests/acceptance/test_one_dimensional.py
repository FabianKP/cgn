"""
Tests ggn_new.solvers.GaussNewton on an unconstrained, nonlinear, one-dimensional problem.
"""

import numpy as np

import cgn
from problem import TestProblem
from do_test import do_test


def beale(x):
    y = (1.5-4.0*x[0])**2 + (2.25-10.0*x[0])**2 + (2.625-28.0*x[0])**2
    return np.array([y])


def bealegrad(x):
    y = 1800.0*(x[0] - 0.113333)
    return np.array([[y]])


class OneDimensionalProblem(TestProblem):
    def __init__(self):
        TestProblem.__init__(self)
        xtruemin = 17 / 150 * np.ones((1, ))
        # make initial uncertainty very high, so that we do not get
        # influence from the regularization term
        x0 = np.array([0.0])
        x = cgn.Parameter(start=x0, name="x")
        # no regularization
        x.beta = 0
        self._problem = cgn.Problem(parameters=[x], fun=beale, jac=bealegrad)
        self._minimum = self._problem.costfun(xtruemin)
        self._tol = 1e-2
        self._options = cgn.Solveroptions()
        self._options.tol = 1e-3


def test_one_dimensional():
    odp = OneDimensionalProblem()
    do_test(odp)



