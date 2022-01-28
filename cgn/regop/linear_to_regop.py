
import numpy as np
from scipy.sparse.linalg import LinearOperator

from .regularization_operator import RegularizationOperator


class RegopFromLinop(RegularizationOperator):
    """
    Constructs a :py:class:`RegularizationOperator` from a :py:class:`LinearOperator` object.
    """
    def __init__(self, a: LinearOperator):
        # Assemble matrix representation
        m, n = a.shape
        self._operator = a
        basis = np.identity(n)
        a_matrix = a.matmat(basis)
        self._mat = a_matrix

    def fwd(self, v: np.ndarray):
        """
        Just executes self._operator.matvec(v).
        """
        return self._operator.matvec(v)

    def adj(self, w: np.ndarray):
        """
        Just executes self._operator.rmatvec(w).
        """
        return self._operator.rmatvec(w)



def linop_to_regop(a: LinearOperator) -> RegularizationOperator:
    """
    Translates a linear operator to a regularization operator.

    :param a: The linear operator of type :py:class:`LinearOperator`.
    :return: An object of type :py:class:`RegularizationOperator`.
    """
    # Check that a is a two-dimensional operator.
    assert a.ndim == 2
    # Create RegopFromLinop from a
    regop = RegopFromLinop(a)
    return regop

