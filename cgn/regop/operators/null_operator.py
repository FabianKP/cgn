
import numpy as np

from ..regularization_operator import RegularizationOperator


class NullOperator(RegularizationOperator):
    """
    The null operator :math:`R(v) = 0`.
    """
    def __init__(self, dim: int):
        mat = np.zeros((1, dim))
        RegularizationOperator.__init__(self, mat)
        self._rdim = 0

    def adj(self, v: np.ndarray):
        """
        Always returns 0.
        """
        return np.zeros((1,))

    def fwd(self, v: np.ndarray):
        """
        Always return 0.
        """
        return np.zeros((1,))