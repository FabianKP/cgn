
import numpy as np
from typing import List, Literal

from .constraint import Constraint
from .parameter import Parameter


class LinearConstraint(Constraint):
    """
    Represents a linear constraint. Either an equality constraint :math:`Ax = b`, or an inequality constraint
    :math:`Ax \\geq b`, where :math:`A \\in \\mathbb{R}^{c \\times n}`.
    """
    def __init__(self, parameters: List[Parameter], a: np.ndarray, b: np.ndarray, ctype: Literal["eq", "ineq"]):
        """
        Represents a linear constraint. Either an equality constraint :math:`Ax = b`, or an inequality constraint
        :math:`Ax \\geq b`, where :math:`A \\in \\mathbb{R}^{c \\times n}`.

        :param parameters: A list of the parameters involved in the constraint. If the list contains more than one
            element, the constraint will be defined with respect to the concatenated parameter vector.
        :param a: Of shape (c, n). The constraint matrix. The number of columns `n` must be equal to the dimension of
            the concatenated parameter vector.
        :param b: The right hand side of the constraint. Must have shape (c,).
        :param ctype: The type of the constraint.
        """
        self._check_input_linear(parameters, a, b, ctype)
        self._cdim = a.shape[0]
        self._a = a
        self._b = b

        def linfun(*args):
            xvec = np.concatenate(args)
            y = self._a @ xvec - b
            return y

        def linjac(*args):
            return self._a
        Constraint.__init__(self, parameters=parameters, fun=linfun, jac=linjac, ctype=ctype)

    @property
    def a(self) -> np.ndarray:
        """
        The constraint matrix :math:`A`.
        """
        return self._a

    @property
    def b(self) -> np.ndarray:
        """
        The constraint vector :math:`b`.
        """
        return self._b

    @property
    def cdim(self) -> int:
        """
        The constraint dimension :math:`c`.
        """
        return self._cdim

    @staticmethod
    def _check_input_linear(parameters: List[Parameter], a: np.ndarray, b: np.ndarray,
                            ctype: Literal["eq", "ineq"]):
        if ctype not in ["eq", "ineq"]:
            raise Exception("'ctype' must either be 'eq' or 'ineq'.")
        n = 0
        for param in parameters:
            n += param.dim
        if not a.shape[1] == n:
            raise Exception(f"'a' must have shape[1] = {n}")
        m = a.shape[0]
        if not b.shape == (m, ):
            raise Exception(f"'b' must have shape ({m},).")
