
import numpy as np

from ..regularization_operator import RegularizationOperator


class IdentityOperator(RegularizationOperator):
    """
    Corresponds to to the identity operator :math:`I(v) = v`.
    """
    def __init__(self, dim):
        self._mat = np.identity(dim)

    def fwd(self, v: np.ndarray) -> np.ndarray:
        """
        See :py:attr:`RegularizationOperator.fwd`.
        """
        return v

    def adj(self, v: np.ndarray) -> np.ndarray:
        """
        See :py:attr:`RegularizationOperator.adj`.
        """
        return v
