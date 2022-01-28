"""
Implement problem 79 from the Hock-Schittkowski collection.
"""


from math import sqrt
import numpy as np

import cgn
from tests.acceptance.do_test import do_test
from tests.acceptance.problem import TestProblem


n = 5
xstart = 2 * np.ones(n)
xmin = np.array( [1.191127, 1.362603, 1.472818, 1.635017, 1.679081] )


def misfit2(x):
    y = np.zeros(5)
    y[0] = x[0]-1.0
    y[1] = x[0]-x[1]
    y[2] = x[1]-x[2]
    y[3] = (x[2]-x[3])**2
    y[4] = (x[3]-x[4])**2
    return y



def misfitjac2(x):
    J = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0*(x[2]-x[3]), 2.0*(x[3]-x[2]), 0.0],
        [0.0, 0.0, 0.0, 2.0*(x[3]-x[4]), 2.0*(x[4]-x[3])]
    ])
    return J



def constraint2(x):
    y = np.zeros(3)
    y[0] = x[0] + x[1]**2 + x[2]**3 - 2 - 3 * sqrt(2)
    y[1] = x[1] - x[2]**2 + x[3] + 2 - 2 * sqrt(2)
    y[2] = x[0]*x[4]-2
    return y



def constraintjac2(x):
    J = np.array([
        [1.0, 2.0*x[1], 3.0*x[2]**2, 0.0, 0.0],
        [0.0, 1.0, -2.0*x[2], 1.0, 0.0],
        [x[4], 0.0, 0.0, 0.0, x[0]]
    ])
    return J


def _compute_starting_value():
    """
    Finds a feasible point x0 s.t. constraintjac2(x0) = 0.
    """
    x = cgn.Parameter(dim=n, name="x")
    x.beta = 1e-6   # small regularization so that problem is well-posed
    solver = cgn.CGN()
    x_init = 2 * np.ones(n)
    problem = cgn.Problem(parameters=[x], fun=constraint2, jac=constraintjac2)
    solution = solver.solve(problem=problem, starting_values=[x_init])
    x0 = solution.minimizer("x")
    return x0


class NonlinearEqualityConstraint2(TestProblem):
    def __init__(self):
        TestProblem.__init__(self)
        self._tol = 1e-5
        x = cgn.Parameter(dim=n, name="x")
        # add inequality constraints
        incon = cgn.NonlinearConstraint(parameters=[x], fun=constraint2, jac=constraintjac2, ctype="eq")
        self._problem = cgn.Problem(parameters=[x], fun=misfit2, jac=misfitjac2, constraints=[incon])
        # Perform nonlinear equation solving to find starting value (using CGN, no less!)
        xstart = _compute_starting_value()
        self._start = [xstart]
        c0 = np.linalg.norm(constraint2(xstart), ord=1)
        # Reduce ctol
        self._options.ctol = c0 * 10.
        # The true solution is known
        self._minimum = self._problem.costfun(xmin)


def test_nonlinear_equality_constraint2():
    op = NonlinearEqualityConstraint2()
    do_test(op)