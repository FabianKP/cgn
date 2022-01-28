
from typing import List, Literal

from .constraint import Constraint
from .parameter import Parameter


class NonlinearConstraint(Constraint):
    """
    Represents a nonlinear constraint. Either an equality constraint :math:`G(x)=0` or an inequality constraint
    :math:`G(x)>=0`.
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
        Constraint.__init__(self, parameters=parameters, fun=fun, jac=jac, ctype=ctype)
