"""
Tests ggn_new.solvers.GaussNewton on an unconstrained linear least-squares problem.
"""

# This software was written by Fabian Parzer.
# Copyright owned by the University of Vienna, 2020. All rights reserved.

import numpy as np
from cgn import *
from sklearn.linear_model import Ridge

from time import time

m = 400
n = 200

# see Mori et al., example (32)
def observationOperator(x):
    y = np.zeros(m)
    for i in range(n):
        y[i] = x[i] - 2.0*sum(x)/float(m) - 1.0
    for i in range(n,m):
        y[i] = - 2.0*sum(x)/float(m)-1.0
    return y

def assembleJacobian():
    A = np.zeros((m,n))
    # upper half of the Jacobian matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i,j] = 1.0-2.0/float(m)
            else:
                A[i,j] = -2.0/float(m)
    # lower half of the Jacobian matrix
    for i in  range(n,m):
        for j in range(n):
            A[i,j] = -2.0/float(m)
    return A



def solveProblem(A,y,b):
    """
    use ridge regression from sklearn as reference _solver
    """
    clf = Ridge(alpha=1.0)
    ymod = y - b    #modify y so that it contains also the bias term
    clf.fit(A,y)
    xdagger = clf.coef_
    return xdagger



def testLinear ():
    delta = 10e-6
    x0 = np.zeros(n)
    A = assembleJacobian()
    b = - np.ones(m) #bias in the linear model, important for ridge regression
    def evaluateJacobian(x):
        return A
    y = np.ones(m)
    def func1(x):
        F = observationOperator(x) - y
        return F

    # Solve with GGN
    t0_cgn = time()
    pars = Parameters()
    pars.addParameter(dim=n, mean=x0)
    linearproblem = MultiParameterProblem(pars, fun=func1, jac=evaluateJacobian)
    cgn = CGN(linearproblem)
    x_cgn = cgn.solve(options={"maxiter": 20, "verbose": True}).minimizer
    t_cgn = time() - t0_cgn

    xcompare = solveProblem(A, y, b)

    def residual(x):
        r = np.linalg.norm(func1(x))**2 + np.linalg.norm(x)**2
        return r
    r_cgn = residual(x_cgn)
    rcompare = residual(xcompare)
    print(f"CGN time: {t_cgn}")
    print(f"CGN residual: {r_cgn}")
    # test succeeded if we reach equally good solution +-10%
    if r_cgn < (1.0 + delta)*rcompare:
        return True
    else:
        return False