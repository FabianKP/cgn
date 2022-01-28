
from copy import deepcopy
import numpy as np


class RegularizationOperator:
    """
    Abstract base class for regularization operators.
    Each child of RegularizationOperator must implement the methods `fwd` and `adj`, which give the forward and the
    adjoint action of the regularization operator.
    """
    _mat: np.ndarray

    @property
    def dim(self) -> int:
        """
        The dimension of the domain of the regularization operator.
        """
        return deepcopy(self._mat.shape[1])

    @property
    def rdim(self) -> int:
        """
        The dimension of the codomain of the regularization operator.
        """
        return deepcopy(self._mat.shape[0])

    @property
    def mat(self) -> np.ndarray:
        """
        The matrix representation of the regularization operator. A matrix of shape (r,n), where
        n= :py:attr:`~dim` and r= :py:attr:`~rdim`.
        """
        return deepcopy(self._mat)

    def fwd(self, v: np.ndarray) -> np.ndarray:
        """
        Evaluates the forward action of the regularization operator.
        :param v: :param v: Of shape (n,m), where n = :py:attr:`~dim`.
        :return: Of shape (r,m), where r = :py:attr:`rdim`.
        """
        raise NotImplementedError

    def adj(self, w: np.ndarray):
        """
        Evaluates the adjoint action of the regularization operator.
        :param w: Of shape (r, m), where r = :py:attr:`~rdim`.
        :return: Of shape (n, m), where n = :py:attr:`~dim`.
        """
        raise NotImplementedError
