"""
Tests ggn_new.solvers.GaussNewton on an unconstrained linear least-squares problem.
"""

# This software was written by Fabian Parzer.
# Copyright owned by the University of Vienna, 2020. All rights reserved.

import numpy as np
from sklearn.linear_model import Ridge

import cgn
from tests.acceptance.problem import TestProblem
from tests.acceptance.do_test import do_test

m = 400
n = 200

# see Mori et al., example (32)
def observation_operator(x):
    y = np.zeros(m)
    for i in range(n):
        y[i] = x[i] - 2.0 * sum(x)/m
    for i in range(n,m):
        y[i] = - 2.0 * sum(x)/m
    return y

def assemble_jacobian():
    A = np.zeros((m,n))
    # upper half of the Jacobian matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i,j] = 1.0 - 2.0 / m
            else:
                A[i,j] = -2.0 / m
    # lower half of the Jacobian matrix
    for i in  range(n,m):
        for j in range(n):
            A[i,j] = -2.0 / m
    return A



def solve_problem(a, y, b):
    """
    use ridge regression from sklearn as reference _solver
    """
    clf = Ridge(alpha=1.0)
    ymod = y - b    #modify y so that it contains also the bias term
    clf.fit(a, ymod)
    xdagger = clf.coef_
    return xdagger


class LinearProblem(TestProblem):
    def __init__(self):
        TestProblem.__init__(self)
        x = cgn.Parameter(start=np.zeros(n), name="x")
        # standard regularization
        x.beta = 1.
        y = np.ones(m)
        b = - np.ones(m)
        j = assemble_jacobian()
        def misfit(x):
            return observation_operator(x) + b - y
        def jac(x):
            return j
        self._problem = cgn.Problem(parameters=[x], fun=misfit, jac=jac)
        # compute actual minimum
        x_ref = solve_problem(a=j, y=y, b=b)
        self._minimizer = x_ref
        self._minimum = self._problem.costfun(x_ref)
        self._tol = 1e-6

def test_linear():
    lp = LinearProblem()
    do_test(lp)
