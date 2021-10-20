
import numpy as np
from typing import Union

from ..regularization_operator import RegularizationOperator


class DiagonalOperator(RegularizationOperator):
    """
    Tool to form diagonal regularization operators of the form :math:`R = \mathrm{diag}(s_1, \ldots, s_n)`.
    """
    def __init__(self, dim, s: Union[float, np.ndarray]):
        """
        :param s: All entries must me larger than 0.
            If s is a float, the resulting diagonal operator will be :math:`R = \mathrm{diag}(s, \ldots, s)`.
            If s is a vector, the resulting diagonal operator will be :math:`R = \mathrm{diag}(s_1, \ldots, s_n)`.
        """
        isfloat = isinstance(s, float)
        if isfloat:
            positive = s > 0
        else:
            positive = np.all(s > 0)
        if not positive:
            raise ValueError("s must be positive.")
        if not isfloat:
            if s.size != dim or s.ndim != 1:
                raise ValueError("s must be float or vector of shape (dim,).")
        self._isfloat = isfloat
        self._s = s
        mat = s * np.identity(dim)
        RegularizationOperator.__init__(self, mat)

    def adj(self, v: np.ndarray):
        """
        See :py:attr:`RegularizationOperator.adj`.
        """
        return self.fwd(v)   # diagonal operators are self-adjoint

    def fwd(self, v: np.ndarray):
        """
        See :py:attr:`RegularizationOperator.fwd`.
        """
        if v.ndim == 1:
            u = self._s * v
        else:
            # matrix case
            if self._isfloat:
                u = self._s * v
            else:
                u = self._s[:, np.newaxis] * v
        return u
