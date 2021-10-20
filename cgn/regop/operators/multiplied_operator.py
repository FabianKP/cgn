"""
Contains class multiplied operator
"""

from copy import deepcopy
import numpy as np

from ..regularization_operator import RegularizationOperator


class MultipliedOperator(RegularizationOperator):
    """
    Implements a regularization operator that is created by right-multiplying a given regularization operator
    with a matrix. That is, given a regularization operator :math`R` and a matrix :math:`Q`, the
    new regularization operator corresponds to :math:`R Q`.
    """
    def __init__(self, regop: RegularizationOperator, q: np.ndarray):
        """

        :param regop: The regularization operator :math:`R`.
        :param q: The matrix :math:`Q` by which the regularization operator is multiplied. It must have shape (n,m),
            where n = :code:`regop.dim`.
        """
        self._op = deepcopy(regop)
        self._q = q.copy()
        mat = regop._mat @ q
        RegularizationOperator.__init__(self, mat)

    def fwd(self, v: np.ndarray):
        """
        See :py:attr:`RegularizationOperator.fwd`.
        """
        u = self._q @ v
        return self._op.fwd(u)

    def adj(self, v: np.ndarray):
        """
        See :py:attr:`RegularizationOperator.adj`.
        """
        # (PQ)^* = Q.T P^*
        return self._q.T @ self._op.adj(v)
