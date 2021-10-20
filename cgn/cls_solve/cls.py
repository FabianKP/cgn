"""
Contains classes "LSIB" and "LSIBType"
"""

from math import sqrt
import numpy as np


class CLS:
    """
    Represents an inequality-constrained least-squares problem.
    min_x ||H x - y||_2^2 / scale
    s.t. A x = b, C x >= d, x >= l.
    """

    def __init__(self, h: np.ndarray, y: np.ndarray,
                 a: np.ndarray = None, b: np.ndarray = None,
                 c: np.ndarray = None, d: np.ndarray = None,
                 l: np.ndarray = None, scale: float = 1.):
        self._check_consistency(h, y, a, b, c, d, l)
        self.h = h / sqrt(scale)
        self.y = y / sqrt(scale)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        if l is not None:
            self.l = l
        else:
            n = self.h.shape[1]
            self.l = -np.inf * np.ones(n)
        # Next, determine the constraint-type of the problem.
        self.equality_constrained = (self.a is not None)
        self.inequality_constrained = (self.c is not None)
        self.bound_constrained = np.isfinite(self.l).any()

    def _check_consistency(self, h, y, a, b, c, d, l):
        m, n = h.shape
        assert y.shape == (m, )
        self._check_constraint(a, b, n)
        self._check_constraint(c, d, n)
        if l is not None:
            assert l.shape == (n, )

    @staticmethod
    def _check_constraint(mat, vec, n):
        if mat is not None:
            assert mat.shape[1] == n
            c = mat.shape[0]
            assert vec.shape == (c, )
        else:
            assert vec is None
