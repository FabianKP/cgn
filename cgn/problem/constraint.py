
import numpy as np
from typing import List, Literal

from .parameter import Parameter


class Constraint:
    """
    Represents an abstract constraint.
    """
    def __init__(self, parameters: List[Parameter], fun: callable, jac: callable, ctype: Literal["eq", "ineq"]):
        """

        :param parameters: A list of the parameters involved in the constraint. If the list contains more than one
            element, the constraint will be defined with respect to the concatenated parameter vector.
        :param fun: The function that determines the constraint. Must take ``len(parameters)`` arguments and return
            a numpy array of shape (c,).
        :param jac: The Jacobian of `fun`. Must take arguments of the same form as `fun`, and return a numpy array
            of shape (c, n), where n is the dimension of the concatenated parameter vector.
        :param ctype: The type of the constraint.
        """
        # Check input
        self._check_consistency(parameters=parameters, fun=fun, jac=jac, ctype=ctype)

        # Compute parameter dimension.
        dim = 0
        for param in parameters:
            dim += param.dim
        self._dim = dim
        self._fun = fun
        self._jac = jac
        # Determine cdim
        testarg = [param.mean for param in parameters]
        y = fun(*testarg)
        self._cdim = y.size
        self._ctype = ctype
        self._parameters = parameters

    def fun(self, *args) -> np.ndarray:
        """
        The constraint function G(x).
        """
        return self._fun(*args)

    def jac(self, *args) -> np.ndarray:
        """
        The constraint jacobian G'(x).
        """
        return self._jac(*args)

    @property
    def ctype(self) -> str:
        """
        The type of the constraint:
            - "eq": equality constraint
            - "ineq": inequality constraint
        """
        return self._ctype

    @property
    def dim(self) -> int:
        """
        The parameter dimension :math:`n`.
        """
        return self._dim

    @property
    def cdim(self) -> int:
        """
        The dimension :math:`c` of the codomain of the constraint function,
        i.e. :math:`G:\\mathbb{R}^n \to \\mathbb{R}^c`.
        """
        return self._cdim

    @property
    def parameters(self) -> List[Parameter]:
        """
        The parameters with respect to which the constraint is defined.
        """
        return self._parameters

    @staticmethod
    def _check_consistency(parameters: List[Parameter], fun: callable, jac: callable, ctype: Literal["eq", "ineq"]):
        """
        Given a feasible vector, checks whether the specifications are consistent.
        """
        if ctype not in ["eq", "ineq"]:
            raise Exception("'ctype' must be either 'eq' or 'ineq'.")
        n = sum([param.dim for param in parameters])
        start_list = [param.start for param in parameters]
        y = fun(*start_list)
        m = y.size
        y_good_shape = y.shape == (m,)
        if not y_good_shape:
            raise Exception(f"The function 'fun' must return numpy arrays of shape ({m}, ).")
        j = jac(*start_list)
        jac_shape_good = j.shape == (m, n)
        if not jac_shape_good:
            raise Exception(f"The function 'jac' must return arrays of shape ({m}, {n}) but return arrays of shape "
                            f"{j.shape}")