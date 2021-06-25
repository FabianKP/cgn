"""
Contains class BNLS.
"""

from regop import RegularizationOperator

from .constraints import Constraint

class CNLS:
    """
    Contained class that manages the user-provided description of the
    constrained nonlinear least-squares problem:
    min_x (0.5*|| func(x) ||^2 + 0.5*alpha*||P(x-xbar)||^2
    s.t. Ax = b, Cx >= d, x >= lb
    Attributes should be self-explaining, except maybe "ctype".
    :attr func:
    :attr jac:
    :attr alpha
    :attr s:
    :attr xbar:
    :attr lb:
    """
    def __init__(self, func, jac, regop: RegularizationOperator, xbar, eqcon: Constraint, incon: Constraint, lb):
        self.func = func
        self.jac = jac
        self.regop = regop
        self.xbar = xbar
        self.dim = xbar.size
        self.eqcon = eqcon
        self.incon = incon
        self.lb = lb

class CNLSSolution:
    """
    The solution of a cnlls problem is an object that has two attributes.
    :attr minimizer: The minimizer of the cnlls problem. A numpy vector.
    :attr precision: The (estimated) posterior precision matrix. A matrix of shape (minimizer.size, minimizer.size).
    """
    def __init__(self, minimizer, precision, min_cost, costfun, info=None):
        assert minimizer.size == precision.shape[0] == precision.shape[1], "Dimensions do not match."
        self.minimizer = minimizer
        self.precision = precision
        self.cost = min_cost
        self.costfun = costfun
        self.info = info