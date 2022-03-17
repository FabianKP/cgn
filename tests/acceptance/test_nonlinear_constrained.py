"""
Test CGN for an equality-constrained nonlinear least-squares problem.
Based on tests (29) in More, Garbow and Hillstrom "Testing Unconstrained Optimization Software"
"""

import numpy as np
import scipy.optimize as sciopt

import cgn
from tests.acceptance.problem import TestProblem
from tests.acceptance.do_test import do_test


n = 300
m = n
h = 1 / (n+1)
t = np.arange(1, n+1) * h


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


class NonlinearConstrainedProblem(TestProblem):

    def __init__(self):
        TestProblem.__init__(self)
        x0 = t * (1 - t)
        x = cgn.Parameter(start=np.zeros(n), name="x")
        x.beta = 0.
        self._problem = cgn.Problem(parameters=[x], fun=dbv, jac=dbv_jac)
        # compute reference solution with scipy.optimize:
        loss_fun = self._problem.costfun
        loss_grad = self._problem.costgrad
        x_ref = sciopt.minimize(fun=loss_fun, jac=loss_grad, x0=x0).x
        self._minimum = loss_fun(x_ref)

        # Make a sum-to-one constraint that holds at the reference solution.
        a = np.ones((1, n))
        b = np.array(a @ x_ref).reshape((1,))
        x0 = x0 / np.sum(x0) * np.sum(x_ref)
        assert np.isclose(a @ x0, b).all()
        x.start = x0 * np.ones(n)
        # Add the constraint to the cgn.Problem object
        eqcon = cgn.LinearConstraint(parameters=[x], a=a, b=b, ctype="eq")
        self._problem.constraints.append(eqcon)
        # Resolve problem with SLSQP
        def eqfun(x):
            return a @ x - b
        def eqjac(x):
            return a
        eqcon = {"type": "eq", "fun": eqfun, "jac": eqjac}
        x_ref = sciopt.minimize(fun=loss_fun, jac=loss_grad, x0=x0, constraints=(eqcon)).x
        self._minimum = self._problem.costfun(x_ref)
        self._tol = 1e-3


def test_nonlinear_constrained():
    ncp = NonlinearConstrainedProblem()
    do_test(ncp)

