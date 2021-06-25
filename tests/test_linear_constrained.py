"""
Tests ggn_new.solvers.GaussNewton on an unconstrained linear least-squares problem.
"""

# This software was written by Fabian Parzer.
# Copyright owned by the University of Vienna, 2020. All rights reserved.

import numpy as np
from cgn import *
from cgn.utils import negative_norm
import tgn as tgn
from sklearn.linear_model import Ridge

from time import time

m = 500
n = 250

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

def test_linear_constrained():
    alpha = 1e-8
    x0 = np.zeros(n)
    A = assembleJacobian()
    b = - np.ones(m) #bias in the linear model, important for ridge regression
    def evaluateJacobian(x):
        return A
    y = np.ones(m)
    def func1(x):
        F = observationOperator(x) - y
        return F

    # add equality constraint
    mat = np.ones((1, n))
    v = np.ones(1)
    x0 = np.random.randn(n)

    # solve with CGN
    pars_constrained = Parameters()
    eq = {"mat": mat, "vec": v}
    pars_constrained.addParameter(dim=n, mean=x0, eq=eq, alpha=alpha)
    lse = MultiParameterProblem(pars_constrained, fun=func1, jac=evaluateJacobian)
    cgn = CGN(lse)
    cgn.set_starting_value(pnumber=0, val=x0)
    cgn_solution = cgn.solve({"verbose": True})
    x_cgn = cgn_solution.minimizer

    # benchmark against GGN
    def eqcon(x):
        return mat @ x - v
    def eqcon_jac(x):
        return mat
    parameters = tgn.Parameters()
    parameters.addParameter(dim=n, reg="l2", mean=x0)
    tgn_problem = tgn.LpCNLLS(params=parameters, misfit=func1, misfitjac=evaluateJacobian, alpha=alpha)
    tgn_problem.addEqualityConstraint(func=eqcon, jac=eqcon_jac)
    ggn = tgn.GGN(tgn_problem)
    ggn.set_starting_value(pnumber=0, val=x0)
    tgn_solution = ggn.solve(options={"verbose": True})
    x_ggn = tgn_solution.minimizer

    rtol = 1e-1
    ctol = 1e-1

    eqcon_error_cgn = np.linalg.norm(mat @ x_cgn - v)
    eqcon_error_ggn = np.linalg.norm(mat @ x_ggn - v)
    print(f"eqcon_error_cgn = {eqcon_error_cgn}")
    print(f"eqcon_error_ggn = {eqcon_error_ggn}")
    cost_cgn = cgn_solution.costfun(x_cgn)
    cost_ggn = cgn_solution.costfun(x_ggn)
    print(f"cgn cost: {cost_cgn}")
    print(f"ggn cost: {cost_ggn}")

    if cost_cgn <= (1 + rtol) * cost_ggn and eqcon_error_cgn <= ctol:
        return True
    else:
        return False


