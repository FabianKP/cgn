
from math import sqrt

import numpy as np

from ..regularization_operator import RegularizationOperator


class ScaledOperator(RegularizationOperator):
    """
    Represents a scaled regularization operator :math:`R = \\alpha P`.
    """
    def __init__(self, regop: RegularizationOperator, alpha: float):
        self._sqrt_a = sqrt(alpha)
        self._inv_sqrt_a = 1 / self._sqrt_a
        self._p = regop
        self._mat = self._sqrt_a * self._p._mat

    def fwd(self, v: np.ndarray):
        """
        See :py:attr:`RegularizationOperator.fwd`.
        """
        return self._sqrt_a * self._p.fwd(v)

    def adj(self, v: np.ndarray):
        """
        See :py:attr:`RegularizationOperator.adj`.
        """
        return self._sqrt_a * self._p.adj(v)
