"""
Contains class CNLS.
"""

from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike

from ..regop import RegularizationOperator
from ..problem.linear_constraint import LinearConstraint


class CNLS:
    """
    Contained class that manages the user-provided description of the
    constrained nonlinear least-squares problem:
    min_x (1/scale) * ( 0.5*||Q func(x) ||^2 + 0.5*||R(x-m)||^2 )
    s.t. Ax = b, Cx >= d, x >= lb.
    The regularization term is optional.
    """
    def __init__(self, func: callable, jac: callable, q: RegularizationOperator, m: ArrayLike, r: RegularizationOperator,
                 eqcon: LinearConstraint, incon: LinearConstraint, lb: ArrayLike, scale: float):
        self._check_input(m, r, eqcon, incon, lb)
        self.func = deepcopy(func)
        self.jac = deepcopy(jac)
        self.q = deepcopy(q)
        self.m = deepcopy(m)
        self.r = deepcopy(r)
        self.scale = scale
        self.dim = m.size
        self.a = deepcopy(eqcon.mat)
        self.b = deepcopy(eqcon.vec)
        self.c = deepcopy(incon.mat)
        self.d = deepcopy(incon.vec)
        self.lb = deepcopy(lb)
        self.equality_constrained = (self.a is not None)
        self.inequality_constrained = (self.c is not None)
        self.bound_constrained = np.isfinite(self.lb).any()

    def satisfies_constraints(self, x, tol=1e-5):
        """
        Assert that given vector satisfies constraints up to a given tolerance.
        The error norm is the l^1-norm
        """
        constraint_error = 0.
        if self.a is not None:
            constraint_error += np.linalg.norm(self.a @ x - self.b)
        if self.c is not None:
            constraint_error += np.linalg.norm((self.c @ x - self.d).clip(max=0.))
        if self.lb is not None:
            constraint_error += np.linalg.norm((x - self.lb).clip(max=0.))
        if constraint_error <= tol:
            return True
        else:
            return False

    @staticmethod
    def _check_input(mean, regop, eqcon, incon, lb):
        n = mean.size
        assert regop is None or regop.dim == n
        assert eqcon.dim == n
        assert incon.dim == n
        assert lb.size == n
