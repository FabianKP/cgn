"""
Contains class "MultiParameterProblem"
"""

from math import sqrt

from regop import ScaledOperator
from .parameter import Parameters
from .utils import is_matrix, is_vector


class MultiParameterProblem:
    """
    """

    def __init__(self, params, fun, jac, delta=1., scaling=1.):
        """
        :param params: bgn.Parameters
        :param fun: function
            A function accepting params.nparams arguments. That is, if params contains 3 parameters,
            then fun will be called with misfit(x,y,z). It should return an ndarray of shape (m,)
        :param jac: function
            The Jacobian corresponding to misfit. It will also be called with params.nparams arguments.
            Make sure that the dimensions match. For example, if params consists of 3 parameters of dimension n, l and r,
            then jac(x,y,z) must return an array of shape (m,n+l+r).
        :attr delta: float > 0
            Misfit term is ||func||_2^2 / delta^2.
        :attr scaling: float > 0
            Whole cost function is scaled with factor scaling, i.e. phi(x) becomes phi_new(x) = scaling * phi(x)
        """
        self._scaling = scaling
        self._scaling_factor = sqrt(self._scaling) / delta
        self.params = self._rescale_params(params)
        self._checkFuncJac(fun, jac)
        self._unscaled_fun = fun
        self._unscaled_jac = jac

    def fun(self, *args):
        return self._scaling_factor * self._unscaled_fun(*args)

    def jac(self, *args):
        return self._scaling_factor * self._unscaled_jac(*args)

    def _checkFuncJac(self, func, jac):
        meanlist = []
        for param in self.params:
            meanlist.append(param.mean)
        means = tuple(meanlist)
        try:
            F = func(*means)
            J = jac(*means)
        except:
            raise Exception("Function and Jacobian do not match parameters.")
        xdim = len(self.params.mean)
        assert is_vector(F), "'F' must be vector"
        assert is_matrix(J), "Every Jacobian must be a twodimensional array."
        assert F.shape[0] == J.shape[0], "The dimension of function output does not match " \
                                         "the first dimension of the Jacobian."
        assert J.shape[1] == xdim, "Second dimension of Jacobian does not match parameter" \
                                   " dimension."

    def _rescale_params(self, params: Parameters):
        """
        Modifies a Parameters object such that the regularization term
        alpha * ||s^(-1)(x-mean)||_2^2 becomes scaling * alpha * ||s^(-1)(x-mean)||_2^2
        :param params: Parameters
        :return: Parameters
        Rescaled Parameters object
        """
        params = params
        for param in params:
            param.regop = ScaledOperator(alpha=self._scaling, p=param.regop)
        return params

class MultiParameterSolution:
    """
    The solution of a lpcnlls problem is an object that has two attributes.
    :attr minimizer: The minimizer of the cnlls problem, either a vector of a list of vectors (in the case of
    multiple parameters).
    :attr precision: The (estimated) posterior precision matrix. A matrix of shape (n,n), where n is the combined
    dimension of all the parameters.
    """
    def __init__(self, minimizer, precision, cost, costfun, info=None):
        self.minimizer = minimizer
        self.precision = precision
        self.cost = cost
        self.costfun = costfun
        self.info = info


