"""
Contains class "Parameter".
"""

from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike

from ..regop import RegularizationOperator, IdentityOperator, NullOperator

EPS = np.finfo(float).eps


class Parameter:
    """
    Container class for a user-defined parameter.
    The prior is determined via a _mean, regop, and the regularization.
    """
    def __init__(self, dim: int, mean: ArrayLike=None, regop: RegularizationOperator=None, beta=1.):
        self._dim = dim
        if mean is None:
            self._mean = np.zeros(dim)
        else:
            self._mean = deepcopy(mean)
        if beta == 0:
            # Regularization is turned off
            self._regop = NullOperator(dim)
        elif regop is None:
            self._regop = IdentityOperator(dim)
        else:
            self._regop = deepcopy(regop)
        assert 0 <= beta
        self._beta = beta
        self._rdim = self._regop.rdim

    @property
    def beta(self):
        return self._beta

    @property
    def dim(self):
        return self._dim

    @property
    def mean(self):
        return self._mean.copy()

    @property
    def regop(self):
        return self._regop

    @property
    def rdim(self):
        return self._rdim



