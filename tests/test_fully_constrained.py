"""
Test CGN for an equality-constrained nonlinear least-squares problem.
Based on test (29) in More, Garbow and Hillstrom "Testing Unconstrained Optimization Software"
"""

# TODO: ARE FUNCTION AND JACOBIAN CORRECT??? DOESN'T SEEM LIKE IT

import numpy as np

from cgn import *
from cgn.utils import negative_norm
import tgn as tgn

n = 300  # todo: if it works, increase the dimension to 300
m = n
h = 1 / (n+1)
t = np.arange(1, n+1) * h


def misfit(x):
    """
    Implements the misfit function
    f_i(x) = x_i + 0.5 * h * ( (1-t_i) * sum_{j=0}^i t_j (x_j + t_j + 1)^3
                                + t_i * sum_{j=i+1}^(n-1) (1-t_j) * (x_j + t_j + 1)^3 ),
    where t_i = i*h, h = 1/(n+1).
    :param x: ndarray of shape (n,)
    :return: ndarray of shape (n,)
    """
    u = np.power(x + t + 1, 3)
    v = t * u
    w = (1-t) * u
    # create vectors y and z with y[i] = sum(v[:i+1]), w[i] = sum(w[i+1:])
    y = np.cumsum(v)
    z = np.cumsum(np.flip(w))   # z_i = sum_{j=0}^i w_{n-j}
    # compute f
    f = np.zeros_like(x)
    for i in range(n):
        f[i] = x[i] + 0.5 * h * ((1-t[i]) * y[i] + t[i] * z[n-i-1])
    # it would probably be more elegant to vectorize this, but I don't have those 5 minutes.
    return f

def misfitjac(x):
    """
    Computes the Jacobian corresponding to the function 'misfit' at x.
    Its entries are
        jac[i,j] = delta_{ij} + 0.5 * h * (1-t_j)*t_i*4(x_i+t_i+1)^2, if i>=j;
        jac[i,j] = 0.5 * h * t_j * (1- t_i) * 3 * (x_i + t_i + 1)^2, if i<j.
    :param x: (n,) ndarray
    :return: (n,n) ndarray
    """
    # the Jacobian is computed column wise
    jac = np.zeros((n, n))
    d_u = 3 * np.power(x + t + 1, 2)
    d_v = t * d_u
    d_w = (1 - t) * d_u
    for j in range(n):
        # case j<=i
        jac[j:, j] = 0.5 * h * (1 - t[j:]) * d_v[j]
        # case j==i, add 1.
        jac[j, j] += 1
        # case j>i
        jac[:j, j] = 0.5 * h * t[:j] * d_w[j]
    return jac


def dbv(x):
    """
    Implements discrete boundary value function
    :param x:
    :return:
    """
    u = np.power(x + t + 1, 3)
    x_patched = np.concatenate((np.zeros(1), x, np.zeros(1)))
    x_less = np.roll(x_patched, 1)[1:-1]  # now x_less[i] = x[i-1]
    x_more = np.roll(x_patched, -1)[1:-1] # now x_more[i] = x[i+1]
    f = 2 * x - x_less - x_more + 0.5 * h**2 * u
    return f


def dbv_jac(x):
    """
    Implements the Jacobian of the discrete boundary value function
    """
    jac = np.zeros((n, n))
    du = 3 * np.power(x+t+1, 2)
    # fill row-wise
    for i in range(n):
        if i>0: jac[i, i-1] = - 1
        if i<n-1: jac[i, i+1] = - 1
        jac[i, i] = 2 + 0.5 * h**2 * du[i]
    return jac


#----------------------------------RUN
def test_fully_constrained():
    # setup problem
    x0 = t * (1-t)
    # first, find the unconstrained optimum
    pars = Parameters()
    # want almost no regularization
    s = 1e-8
    pars.addParameter(dim=n, alpha=s)
    test_problem = MultiParameterProblem(params=pars, fun=dbv, jac=dbv_jac)
    gn = CGN(test_problem)
    gn.set_starting_value(0, x0)
    solution = gn.solve(options={"verbose": True})
    x_min = solution.minimizer

    # make a sum-constraint
    b = np.array(np.sum(x_min)).reshape((1,))
    A = np.ones((1, n))
    # make a lower bound constraint
    C = np.identity(n)
    d = x_min

    # solve with CGN
    pars2 = Parameters()
    eq = {"mat": A, "vec": b}
    ineq = {"mat": C, "vec": d}
    pars2.addParameter(dim=n, alpha=s, eq=eq, ineq=ineq)
    constrained_problem = MultiParameterProblem(params=pars2, fun=misfit, jac=misfitjac)
    cgn = CGN(constrained_problem)
    # rescale x0 so that it satisfies the constraint
    cgn.set_starting_value(0, x0)
    constrained_solution = cgn.solve(options={"verbose": True})
    x_cgn = constrained_solution.minimizer

    # benchmark against TGN
    def eqcon(x):
        return A @ x - b
    def eqcon_jac(x):
        return A
    def incon(x):
        return C @ x -d
    def incon_jac(x):
        return C
    tgn_pars = tgn.Parameters()
    tgn_pars.addParameter(dim=n, reg="l2")
    tgn_prob = tgn.LpCNLLS(params=tgn_pars, misfit=misfit, misfitjac=misfitjac, alpha=s)
    tgn_prob.addEqualityConstraint(func=eqcon, jac=eqcon_jac)
    tgn_prob.addInequalityConstraint(func=incon, jac=incon_jac)
    ggn = tgn.GGN(problem=tgn_prob)
    ggn.set_starting_value(0, x0)
    ggn_solution = ggn.solve(options={"verbose": True})
    x_ggn = ggn_solution.minimizer

    rtol = 1e-1
    ctol = 1e-1

    eqcon_error_cgn = np.linalg.norm(A @ x_cgn - b)
    eqcon_error_ggn = np.linalg.norm(A @ x_ggn - b)
    incon_error_cgn = negative_norm(C @ x_cgn - d)
    incon_error_ggn = negative_norm(C @ x_ggn - d)
    print(f"eqcon_error_cgn = {eqcon_error_cgn}")
    print(f"eqcon_error_ggn = {eqcon_error_ggn}")
    print(f"incon_error_cgn = {incon_error_cgn}")
    print(f"incon_error_ggn = {incon_error_ggn}")
    cost_cgn = constrained_solution.costfun(x_cgn)
    cost_ggn = constrained_solution.costfun(x_ggn)
    print(f"cgn cost: {cost_cgn}")
    print(f"ggn cost: {cost_ggn}")

    if cost_cgn <= (1+rtol)*cost_ggn and eqcon_error_cgn <= ctol and incon_error_cgn <= ctol:
        return True
    else:
        return False

