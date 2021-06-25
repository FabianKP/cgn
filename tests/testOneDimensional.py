"""
Tests ggn_new.solvers.GaussNewton on an unconstrained, nonlinear, one-dimensional problem.
"""

# This software was written by Fabian Parzer.
# Copyright owned by the University of Vienna, 2020. All rights reserved.

import numpy as np
from time import time

from cgn import *

# this function has a global minimum at x = 17/150
def beale(x):
    y = (1.5-4.0*x[0])**2 + (2.25-10.0*x[0])**2 + (2.625-28.0*x[0])**2
    return np.array([y])

def bealegrad(x):
    y = 1800.0*(x[0] - 0.113333)
    return np.array([[y]])

def testOneDimensional ():
    tol = 10e-7   #margin of error
    xtruemin = 17 / 150
    Sigma = DiagonalOperator(dim=1, s=1e-3)   # want (1,1) array
    # make initial uncertainty very high, so that we do not get
    # influence from the regularization term
    x0 = np.array([0.0])

    # test GGN
    t0_ggn = time()
    params = Parameters()
    params.addParameter(dim=1, mean=x0, regop=Sigma)
    onedimProblem = MultiParameterProblem(params, fun=beale, jac=bealegrad)
    cgn = CGN(onedimProblem)
    cgn.set_starting_value(0, val=x0)
    options = {"maxiter": 50, "verbose": True, "lsiter": 50}
    xmin_cgn = cgn.solve(options).minimizer
    e_cgn = abs(xmin_cgn - xtruemin)
    t_cgn = time() - t0_ggn

    print(f"CGN time: {t_cgn}")
    print(f"CGN-error: {e_cgn}")

    if e_cgn < tol:
        return True
    else:
        return False