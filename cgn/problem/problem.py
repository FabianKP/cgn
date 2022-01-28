"""
Contains class "MultiParameterProblem"
"""

from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Tuple, Union

from ..regop import RegularizationOperator, IdentityOperator, MatrixOperator
from .constraint import Constraint
from .parameter import Parameter


class Problem:
    """
    Class for formulating regularized nonlinear least-squares problems with linear constraints:

    .. math::
        \\min_{x_1,...,x_p} \\quad & ||Q F(x_1,...,x_p)||_2^2 + \\beta_1 * ||R_1(x_1 - m_1)||_2^2 + \\ldots +
        \\beta_2 * ||R_p(x_p - m_p)||_2^2 \\\\
          s.t. \\quad & Ax = b, \\quad Cx \\geq d, G(x) = 0, H(x) \\geq 0, \\quad l \\leq x \\leq u.

    :ivar m: The dimension of the codomain of the function :math:``F``.
    :ivar n: The dimension of the concatenated parameter vector :math:``x = (x_1, x_2, ..., x_p)``.
    """
    def __init__(self, parameters: List[Parameter], fun: callable, jac: callable,
                 q: Union[np.ndarray, RegularizationOperator] = None, constraints: List[Constraint] = None,
                 scale: float = 1.):
        """
        :param parameters: The parameters on which the problem depends, e.g. [x1, x2, ..., xp].
        :param fun: A function accepting arguments corresponding to ``parameters``.
            For example, if ``parameters = [x, y, z]``, then ``fun(u, v, w)`` should be defined, where e.g. v would
             be a numpy array of shape (y.dim,). The output of ``fun(u, v, w)`` should be a numpy array of shape (m,).
        :param jac: The Jacobian corresponding to `fun`. It should accept the same argument as `fun`, and the output
            should be a numpy array with shape (m, n), where n is the sum of the dimensions of the parameters.
        :param q: The regularization of the misfit term. Typically, this will be a square root of the noise precision
            matrix. Can either be a numpy array or a :py:class:`RegularizationOperator`.
        :param scale: The scale of the cost function. This only matters for the optimization.
            If provided, the cost function is divided by the scale.
            A good default choice for this parameter is m.
        """
        self._check_input(parameters, fun, jac, q, constraints, scale)
        self._parameter_list = parameters
        # Initialize constraints
        if constraints is None:
            self._constraints = []
        else:
            self._constraints = constraints
        self._nparams = len(parameters)
        # Set the shape-attribute
        self._shape = self._determine_shape(parameters)
        # get measurement dimension m
        self.m, self.n = self._find_m_n(parameters, fun)
        self.scale = scale
        self.q = self._default_regop(q, self.n)
        self.fun = deepcopy(fun)
        self.jac = deepcopy(jac)

    @property
    def nparams(self) -> int:
        """
        :return: The number of parameters on which the problem depends.
        """
        return self._nparams

    @property
    def shape(self) -> Tuple[int]:
        """
        The shape of the problem. For example, if the problem depends on 3 parameters of dimensions 3, 5, 7,
        then ``Problem.shape`` returns a tuple (3, 5, 7).
        """
        return self._shape

    @property
    def constraints(self) -> List[Constraint]:
        """
        The constraints given to the problem in initialization.
        """
        return self._constraints

    @constraints.setter
    def constraints(self, value: List[Constraint]):
        self._check_constraints(value, self._parameter_list)
        self._constraints = value

    def costfun(self, *args) -> float:
        """
        Returns the cost function.

        .. math::
            \\phi(x_1,\\ldots, x_p) = \\frac{1}{2} \\left( ||Q F(x_1,...,x_p)||_2^2 + \\beta_1 ||R_1(x_1 - m_1)||_2^2 +
            \\ldots + \\beta_p ||R_p(x_p - m_p)||_2^2 \\right).

        :param args: The number of args should be equal to :py:attr:`~nparams`.
        :return: Value of the cost function at given parameters.
        """
        misfit = 0.5 * np.sum(np.square(self.q.fwd(self.fun(*args))))
        regularization = 0
        for param, arg in zip(self._parameter_list, args):
            reg = 0.5 * param.beta * np.sum(np.square(param.regop.fwd(arg - param.mean)))
            regularization += reg
        return misfit + regularization

    def costgrad(self, *args) -> np.ndarray:
        """
        Returns the gradient of the cost function:

        .. math::
            \\nabla \\phi(x_1,\\ldots,x_p) = (QF'(x_1,\\ldots,x_p))^\\top Q F(x_1,\\ldots,x_p) +
            \\beta_1 R_1^\\top R_1 (x_1 - m_1) + \\ldots + \\beta_p R_p^\\top R_p (x_p - m_p).

        :param args: The number of args should be equal to :py:attr:`~nparams`.
        :returns: :math:`\\nabla \\phi`, of shape (n,).
        """
        misfit_grad = (self.q.fwd(self.jac(*args))).T @ self.q.fwd(self.fun(*args))
        reg_grad_list = []
        for param, arg in zip(self._parameter_list, args):
            reg_grad = param.beta * param.regop.adj(param.regop.fwd(arg - param.mean))
            reg_grad_list.append(reg_grad)
        regularization_grad = np.concatenate(reg_grad_list)
        return misfit_grad + regularization_grad

    def parameter(self, pname: str) -> Parameter:
        """
        Gives access to the parameter of the given name.
        :raises Exception: If no parameter of name ``pname`` is found.
        """
        for param in self._parameter_list:
            if param.name == pname:
                return param
        raise Exception(f"No parameter with name {pname} found.")

    @property
    def parameters(self) -> Tuple[Parameter]:
        """
        The parameters of the optimization problem.
        """
        return tuple(self._parameter_list)

    # PROTECTED

    def _check_input(self, parameters: List[Parameter], fun: callable, jac: callable,
                     q: Union[np.ndarray, RegularizationOperator],
                     constraints: Union[List[Constraint], None], scale: float):
        # Check that no two parameters have the same name
        self._no_duplicate_names(parameters)
        # Check that fun and jac can take arguments that are of the form suggested by dims
        m, n = self._find_m_n(parameters, fun)
        #   Check that the dimensions of the output of jac match fun and the parameter dimension
        x_list = []
        for param in parameters:
            x_list.append(np.zeros(param.dim))
        j = jac(*x_list)
        if j.shape != (m, n):
            raise Exception("Dimensions are inconsistent.")
        # Check that regop is either None or ArrayLike or RegularizationOperator, and of right dimension
        self._check_regop(q, m)
        # Check that the constraints are defined with respect to the given parameters.
        if constraints is not None:
            self._check_constraints(parameters=parameters, constraints=constraints)
        # scale must be positive
        if scale <= 0:
            raise Exception("'scale' must be positive.")

    @staticmethod
    def _check_constraints(constraints: List[Constraint], parameters: List[Parameter]):
        # For each constraint, check that it only depends on parameters in 'parameters'
        for constraint in constraints:
            # Get list of parameters the constraint depends on.
            constraint_parameters = constraint.parameters
            # Check that this list is a subset of ``parameters``.
            if not set(constraint_parameters).issubset(set(parameters)):
                raise Exception("The constraints in 'constraints' are only allowed to depend on parameters from "
                                "'parameters'.")

    @staticmethod
    def _default_regop( regop: Union[ArrayLike, RegularizationOperator, None], dim: int) -> RegularizationOperator:
        if regop is None:
            # default to identity
            regop = IdentityOperator(dim=dim)
        elif regop is np.ndarray:
            if not regop.ndim == 2:
                raise ValueError("'s' must be a 2-dimensional numpy array.")
            else:
                regop = MatrixOperator(mat=regop)
        else:
            regop = deepcopy(regop)
        return regop

    @staticmethod
    def _determine_shape(parameters: List[Parameter]) -> Tuple[int]:
        """
        Determines the shape.
        """
        shape_list = []
        for param in parameters:
            shape_list.append(param.dim)
        return tuple(shape_list)

    @staticmethod
    def _check_regop(regop, dim):
        if regop is not None:
            if isinstance(regop, RegularizationOperator):
                if regop.dim != dim:
                    raise ValueError(f"regop.dim must equal {dim}")
            else:
                # regop must be array, or the input is invalid.
                if regop.shape != (dim, dim):
                    raise ValueError(f"regop must be a matrix of shape ({dim}, {dim})")
        else:
            return True

    @staticmethod
    def _find_m_n(parameters: List[Parameter], fun: callable):
        x_list = []
        n = 0
        for param in parameters:
            x_list.append(np.zeros(param.dim))
            n += param.dim
        y = fun(*x_list)
        m = y.size
        return m, n

    @staticmethod
    def _no_duplicate_names(parameters: List[Parameter]):
        """
        Checks that no two parameters have the same name.

        :raises Exception: If a duplicate is found.
        """
        # Get list of names
        name_list = []
        for param in parameters:
            name_list.append(param.name)
        # Check for duplicates
        contains_duplicates = len(name_list) != len(set(name_list))
        if contains_duplicates:
            raise Exception("'parameters' contains duplicate names.")
