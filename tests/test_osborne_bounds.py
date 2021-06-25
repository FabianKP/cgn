"""
Tests ggn_new.solvers.GeneralizedGaussNewton on the Osborne17 problem (see Mori et al. )
"""

# This software was written by Fabian Parzer.
# Copyright owned by the University of Vienna, 2020. All rights reserved.

from math import exp
import numpy as np

from time import time

from cgn import *
from regop import *

m = 65
n = 11



def observationpre(x, t):
    y = np.ones(m)
    for i in range(m):
            y[i] = x[0]*exp(-x[4]*t[i]) + x[1]*exp(-x[5]*((t[i]-x[8])**2)) \
                + x[2]*exp(-x[6]*((t[i]-x[9])**2)) + x[3]*exp(-x[7]*((t[i]-x[10])**2))
    return y



def observationJanpre(x, t):
    J = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            J[i,0] = exp(-x[4]*t[i])
            J[i,1] = exp(-x[5]*(t[i]-x[8])**2)
            J[i,2] = exp(-x[7]*(t[i]-x[9])**2)
            J[i,3] = exp(-x[7]*(t[i]-x[10])**2)
            J[i,4] = - t[i]*x[0]*exp(-x[4]*t[i])
            J[i,5] = - (t[i]-x[8])**2 * x[1]*exp(-x[5]*(t[i]-x[8])**2)
            J[i,6] = - (t[i]-x[9])**2 * x[2]*exp(-x[6]*(t[i]-x[9])**2)
            J[i,7] = - (t[i]-x[10])**2 * x[3]*exp(-x[7]*(t[i]-x[10])**2)
            J[i,8] = 2.0*x[5]*(t[i]-x[8])*x[1]*exp(-x[5]*(t[i]-x[8])**2)
            J[i,9] = 2.0*x[6]*(t[i]-x[9])*x[2]*exp(-x[6]*(t[i]-x[9])**2)
            J[i,10] = 2.0**x[7]*(t[i]-x[10])*x[3]*exp(-x[7]*(t[i]-x[10])**2)
    return J



def generateParameters():
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
    return y, t



def test_osborne_bounds():
    rtol = 0.01
    sigma = 1e-5
    p = DiagonalOperator(dim=n, s=sigma)
    y, t = generateParameters()
    def observation(x):
        return observationpre(x,t)
    def observationJac(x):
        return observationJanpre(x,t)
    def misfit(x):
        return observation(x)-y
    def cost(x):
        return 0.5*np.linalg.norm(misfit(x))**2 + 0.5*np.linalg.norm(p.fwd(x))**2
    def costJac(x):
        return misfit(x).T @ observationJac(x) + x.T
    x0 = np.zeros(n)
    xstart = np.array([1.3, 0.65, 0.65, 0.7, 0.6, 3.0, 5.0, 7.0, 2.0, 4.5, 4.5])
    ropt = 4.01377e-2

    t0_cgn = time()
    pars = Parameters()
    ineq = {"mat": np.identity(n), "vec": np.zeros(n)}
    lb = np.zeros(n)
    pars.addParameter(dim=n, mean=x0, regop=p, lb=lb)
    # setup problem
    osborneProblem = MultiParameterProblem(
        params=pars,
        fun=misfit,
        jac=observationJac
    )
    cgn = CGN(osborneProblem)
    cgn.set_starting_value(pnumber=0, val=xstart)
    x_cgn = cgn.solve(options={"verbose": True}).minimizer
    t_cgn = time()-t0_cgn

    print(f"x_cgn: {x_cgn}")
    r_cgn = cost(x_cgn)
    print(f"CGN time: {t_cgn}")
    print(f"CGN residual: {r_cgn}")
    print(f"Optimal residual: {ropt}")
    if r_cgn < (1+rtol)*ropt:
        return True
    else:
        return False






