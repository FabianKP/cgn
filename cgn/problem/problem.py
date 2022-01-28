"""
Contains class "MultiParameterProblem"
"""

from copy import deepcopy
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union

from ..regop import RegularizationOperator, IdentityOperator, MatrixOperator
from .linear_constraint import LinearConstraint
from .parameters import Parameters


class Problem:
    """
    Class for formulating regularized nonlinear least-squares problems with linear constraints:

    .. math::
        \\min_{x_1,...,x_p} \\quad & ||Q F(x_1,...,x_p)||_2^2 + \\beta_1 * ||R_1(x_1 - m_1)||_2^2 + \ldots +
        \\beta_2 * ||R_p(x_p - m_p)||_2^2 \\\\
          s.t. \\quad & Ax = b, \\quad Cx \geq d, \quad x \geq l.
    """
    def __init__(self, dims: List[int], fun: callable, jac: callable,
                 q: Union[np.ndarray, RegularizationOperator] = None,
                 scale: float = 1.):
        """
        :param dims: The dimensions of the parameters.
        :param fun: A function accepting argument of the format specified by `dims`. That is, it should accept
            `len(dims)` arguments, and the i-th argument must be a numpy array with shape `(dims[i],)`. The output
            should be a numpy array of shape (m,).
        :param jac: The Jacobian corresponding to `fun`. It should accept the same argument as `fun`, and the output
            should be a numpy array with shape (m, n), where n = `sum(dims)`.
        :param q: The regularization of the misfit term. Typcially, this will be a square root of the noise precision
            matrix. Can either be a numpy array or a :py:class:`RegularizationOperator`.
        :param scale: The scale of the cost function. This only matters for the optimization.
            If provided, the cost function is divided by the scale.
            A good default choice for this parameter is m.
        """
        self._check_input(dims, fun, jac, scale, q)
        self._dims = dims
        self._nparams = len(dims)
        # get measurement dimension m
        self.m, self.n = self._find_m_n(dims, fun)
        self.scale = scale
        self.q = self._default_regop(q, self.n)
        # initialize parameters
        self._params = Parameters(dims)
        self.fun = deepcopy(fun)
        self.jac = deepcopy(jac)
        # initialize constraints
        self._lower_bound = - np.inf * np.ones(self.n)
        self._equality_constraints = []
        self._inequality_constraints = []

    @property
    def nparams(self) -> int:
        """
        :return: The number of parameters on which the problem depends.
        """
        return self._nparams

    @property
    def equality_constraint(self) -> LinearConstraint:
        """
        :return: The equality constraint as :py:class:`LinearConstraint` object.
        """
        # first, combine the constraints
        overall_equality_constraint = self._combine_constraints(self._equality_constraints)
        return deepcopy(overall_equality_constraint)

    @property
    def inequality_constraints(self) -> LinearConstraint:
        """
        :return: The inequality constraint as :py:class:`LinearConstraint` object.
        """
        overall_inequality_constraint = self._combine_constraints(self._inequality_constraints)
        return deepcopy(overall_inequality_constraint)

    @property
    def lower_bound(self):
        """
        :return: The current lower bound `l`, where :math:`x \geq l`. If a component is not lower bounded,
            the corresponding value will be `-np.inf`.
        """
        return deepcopy(self._lower_bound)

    @property
    def parameters(self):
        return deepcopy(self._params)

    def costfun(self, *args) -> float:
        """
        Returns the cost function.

        .. math::
            \phi(x_1,\ldots, x_p) = \\frac{1}{2} \\left( ||Q F(x_1,...,x_p)||_2^2 + \\beta_1 ||R_1(x_1 - m_1)||_2^2 +
            \ldots + \\beta_p ||R_p(x_p - m_p)||_2^2 \\right).

        :param args: The number of args should be equal to :py:attr:`~nparams`.
        :return: Value of the cost function at given parameters.
        """
        misfit = 0.5 * np.sum(np.square(self.q.fwd(self.fun(*args))))
        regularization = 0
        for param, arg in zip(self._params.list, args):
            reg = 0.5 * param.beta * np.sum(np.square(param.regop.fwd(arg - param.mean)))
            regularization += reg
        return misfit + regularization

    def costgrad(self, *args) -> np.ndarray:
        """
        Returns the gradient of the cost function:

        .. math::
            \\nabla \phi(x_1,\ldots,x_p) = (QF'(x_1,\ldots,x_p))^\\top Q F(x_1,\ldots,x_p) +
            \\beta_1 R_1^\\top R_1 (x_1 - m_1) + \ldots + \\beta_p R_p^\\top R_p (x_p - m_p).

        :param args: The number of args should be equal to :py:attr:`~nparams`.
        :returns: :math:`\\nabla \phi`, of shape (n,).
        """
        misfit_grad = (self.q.fwd(self.jac(*args))).T @ self.q.fwd(self.fun(*args))
        reg_grad_list = []
        for param, arg in zip(self._params.list, args):
            reg_grad = param.beta * param.regop.adj(param.regop.fwd(arg - param.mean))
            reg_grad_list.append(reg_grad)
        regularization_grad = np.concatenate(reg_grad_list)
        return misfit_grad + regularization_grad

    def add_equality_constraint(self, a: np.ndarray, b: np.ndarray, i: int = None):
        """
        Adds an equality constraint of the form :math:`A x = b`.

        :param a: Of shape (c,n), where n must equal the dimension of the desired parameter.
        :param b: Of shape (c,).
        :param i: The number of the parameter for which the constraint is set. For example, if `paramno=2`, then the
            constraint will be :math:`A x_2 = b`. If `paramno` is not specified, a global constraint of the
            form :math:`A x = b` is set, where `x` is the concatenated parameter vector.
        """
        self._check_constraint(a, b, i)
        a2 = a.copy()
        if i is not None:
            a2 = self._enlarge_constraint_matrix(a, i)
        constraint = LinearConstraint(dim=self.n, mat=a2, vec=b)
        self._equality_constraints.append(constraint)

    def add_inequality_constraint(self, c: ArrayLike, d: ArrayLike, i: int = None):
        """
        Adds an inequality constraint of the form :math:`Cx \geq d`.

        :param c: Of shape (c,n), where n must equal the dimension of the desired parameter.
        :param d: Of shape (c,).
        :param i: The number of the parameter for which the constraint is set. For example, if `paramno=2`,
            then the constraint will be :math:`C x_2 \geq d`. If `paramno` is not specified, a global constraint of the
            form :math:`C x \geq d` is set, where `x` is the concatenated parameter vector.
        """
        self._check_constraint(c, d, i)
        c2 = c.copy()
        if i is not None:
            c2 = self._enlarge_constraint_matrix(c, i)
        constraint = LinearConstraint(dim=self.n, mat=c2, vec=d)
        self._inequality_constraints.append(constraint)

    def set_lower_bound(self, lb: np.ndarray, i: int = None):
        """
        Adds a lower bound constraint of the form :math:`x_i \geq l`.

        :param lb: The lower bound. Must be of shape (n,), where n is the dimension of the corresponding parameter.
        :param i: The number of the parameter for which the lower bound is set.
        """
        if i is None:
            paramdim = self.n
        else:
            if not 0 <= i < self._nparams:
                raise ValueError(f"i must lie between 0 and {self._nparams}-1")
            paramdim = self._dims[i]
        if lb.shape != (paramdim,):
            raise ValueError(f"lb must have shape ({paramdim},).")
        if not np.all(lb < np.inf):
            raise ValueError(f"A positively infinite lower bound is not allowed.")
        # Update lower bound.
        i0 = self._params.position(i)
        self._lower_bound[i0:i0 + paramdim] = lb

    def set_regularization(self, i: int, m: np.ndarray = None, beta: float = 1.,
                           r: Union[RegularizationOperator, np.ndarray, None] = None):
        """
        Specify a regularization term for the desired parameter. The regularization term will be of the form
        :math:`\\beta * ||R (x - m)||_2^2`.
        
        :param i: Number of the parameter for which you want to specify a regularization term.
        :param m: Of shape (n,). If not provided, defaults to zero.
        :param beta: The regularization parameter. Defaults to 1. You can turn off regularization by setting beta = 0.
        :param r: The regularization operator for the parameter. Defaults to the identity.
        :raises ValueError: If the input is inconsistent.
        """
        # check that paramno is valid
        assert 0 <= i < self._nparams
        # check that input is of the right dimension
        dim = self._dims[i]
        if m is not None:
            if m.size != dim:
                raise ValueError(f"'mean' must have dimension {dim}")
        regop = self._default_regop(r, self.n)
        # modify parameter
        self._params.change_parameter(paramno=i, mean=m, regop=regop, beta=beta)

    # PROTECTED

    def _check_constraint(self, a, b, paramno):
        # check that dimensions are correct
        if paramno is None:
            n = self.n
        else:
            n = self._dims[paramno]
        c = b.size
        if a.shape != (c, n):
            raise ValueError(f"a must have shape ({c},{n})")

    def _check_input(self, dims, fun, jac, alpha, regop):
        # check that fun and jac can take arguments that are of the form suggested by dims
        m, n = self._find_m_n(dims, fun)
        # check that the dimensions of the output of jac match fun and the parameter dimension
        x_list = []
        for dim in dims:
            x_list.append(np.zeros(dim))
        j = jac(*x_list)
        if j.shape != (m, n):
            raise ValueError("Dimensions are inconsistent.")
        # alpha must be positive
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        # check that regop is either None or ArrayLike or RegularizationOperator, and of right dimension
        self._check_regop(regop, m)

    def _combine_constraints(self, constraint_list: List[LinearConstraint]):
        """
        Given a list of constraints, returns a single constraint.
        :param constraint_list: List of objects of type constraint
        :return:
        """
        # If constraint_list is empty, return an empty constraint
        if len(constraint_list) == 0:
            combined_constraint = LinearConstraint(dim=self.n)
        else:
            # create list of all constraint matrices and vectors
            mat_list = []
            vec_list = []
            for constraint in constraint_list:
                mat = constraint.mat
                vec = constraint.vec
                assert mat.shape[1] == self.n
                assert vec.size == mat.shape[0]
                mat_list.append(constraint.mat)
                vec_list.append(constraint.vec)
                assert constraint.dim == self.n
            # concatenate constraints
            combined_mat = np.concatenate(mat_list, axis=0)
            combined_vec = np.concatenate(vec_list, axis=0)
            combined_constraint = LinearConstraint(dim=self.n, mat=combined_mat, vec=combined_vec)
        return combined_constraint

    def _default_regop(self, regop: Union[ArrayLike, RegularizationOperator, None], dim: int) -> RegularizationOperator:
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

    def _enlarge_constraint_matrix(self, a, paramno):
        # rewrite the constraint so that it is defined on the concatenated parameter vector
        c = a.shape[0]
        a2 = np.zeros((c, self.n))
        i0 = self.parameters.position(paramno)
        d = a.shape[1]
        a2[:, i0:i0 + d] = a
        return a2

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
    def _find_m_n(dims, fun):
        x_list = []
        n = 0
        for dim in dims:
            x_list.append(np.zeros(dim))
            n += dim
        y = fun(*x_list)
        m = y.size
        return m, n
