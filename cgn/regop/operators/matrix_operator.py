
import numpy as np

from ..regularization_operator import RegularizationOperator


class MatrixOperator(RegularizationOperator):
    """
    Builds regularization operator from matrix.
    """
    def __init__(self, mat: np.ndarray):
        RegularizationOperator.__init__(self, mat)

    def adj(self, v: np.ndarray) -> np.ndarray:
        """
        See :py:attr:`RegularizationOperator.adj`.
        """
        return self._mat.T @ v

    def fwd(self, v: np.ndarray) -> np.ndarray:
        """
        See :py:attr:`RegularizationOperator.fwd`.
        """
        return self._mat @ v