"""
Contains class CNLS.
"""

from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike

from ..regop import RegularizationOperator
from .cnls_constraint import CNLSConstraint, NullConstraint


class CNLS:
    """
    Contained class that manages the user-provided description of the
    constrained nonlinear least-squares problem:

    .. math::
        min_x (1/scale) * ( 0.5*||Q F(x) ||^2 + 0.5*||R(x-m)||^2 )
        s.t. G(x) = 0, H(x) \\geq 0, l \\leq x \\leq u.
    The regularization term is optional.
    """
    def __init__(self, func: callable, jac: callable, q: RegularizationOperator, m: ArrayLike, r: RegularizationOperator,
                 eqcon: CNLSConstraint, incon: CNLSConstraint, lb: ArrayLike, ub: ArrayLike, scale: float):
        self._check_input(m, r, eqcon, incon, lb, ub)
        self.func = deepcopy(func)
        self.jac = deepcopy(jac)
        self.q = deepcopy(q)
        self.m = deepcopy(m)
        self.r = deepcopy(r)
        self.scale = scale
        self.dim = m.size
        self.g = eqcon.fun
        self.g_jac = eqcon.jac
        self.h = incon.fun
        self.h_jac = incon.jac
        self.lb = deepcopy(lb)
        self.ub = deepcopy(ub)
        self.equality_constrained = not isinstance(eqcon, NullConstraint)
        self.inequality_constrained = not isinstance(incon, NullConstraint)
        self.bound_constrained = np.isfinite(self.lb).any() or np.isfinite(self.ub).any()

    def constraint_violation(self, x: np.ndarray) -> float:
        """
        Computes the constraint violation
            cv = ||(ub - x)^-||_1 + ||(x - lb)^-||_1 + ||g(x)||_1 + ||h(x)^-||_1.
        :param x:
        :return:
        """
        constraint_error = 0.
        if self.equality_constrained:
            constraint_error += np.linalg.norm(self.g(x))
        if self.inequality_constrained:
            constraint_error += np.linalg.norm((self.h(x)).clip(max=0.))
        if self.bound_constrained:
            constraint_error += np.linalg.norm((self.lb - x).clip(min=0.))
            constraint_error += np.linalg.norm((x - self.ub).clip(min=0.))
        return constraint_error

    def satisfies_constraints(self, x, tol=1e-5):
        """
        Assert that given vector satisfies constraints up to a given tolerance.
        The error norm is the l^1-norm
        """
        constraint_error = self.constraint_violation(x)
        if constraint_error <= tol:
            return True
        else:
            print(f"Constraint error to high ({constraint_error} / {tol}).")
            return False

    @staticmethod
    def _check_input(mean, regop, eqcon, incon, lb, ub):
        n = mean.size
        assert regop is None or regop.dim == n
        assert eqcon.dim == n
        assert incon.dim == n
        assert lb.size == n
        assert ub.size == n
