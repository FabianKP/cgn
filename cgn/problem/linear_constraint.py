
from copy import deepcopy
import numpy as np
from typing import Union


class LinearConstraint:
    """
    Represents a linear constraint. Either an equality constraint :math:`Ax = b`, or an inequality constraint
    :math:`Ax \geq b`, where :math:`A \in \mathbb{R}^{c \times n}.
    """
    def __init__(self, dim: int, mat: np.ndarray = None, vec: np.ndarray = None):
        if mat is None:
            # empty constraint
            self._dim = dim
            self._cdim = 0
            self._mat = None
            self._vec = None
            self._empty = True
        else:
            assert mat.shape[1] == dim
            self._dim = dim
            self._cdim = mat.shape[0]
            self._mat = deepcopy(mat)
            self._vec = deepcopy(vec)
            self._empty = False

    @property
    def mat(self) -> Union[np.ndarray, None]:
        """
        The constraint matrix :math:`A`.
        """
        if self._mat is None:
            return None
        else:
            return self._mat.copy()

    @property
    def vec(self) -> Union[np.ndarray, None]:
        """
        The constraint vector :math:`b`.
        """
        if self._vec is None:
            return None
        else:
            return self._vec.copy()

    @property
    def dim(self) -> int:
        """
        The parameter dimension :math:`n`.
        """
        return self._dim

    @property
    def cdim(self) -> int:
        """
        The constraint dimension :math:`c`.
        """
        return self._cdim

    @property
    def empty(self) -> bool:
        """
        `True`, if the constraint is empty. Otherwise `False`.
        """
        return self._empty



