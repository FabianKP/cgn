"""
Contains classes Parameter and Parameters.
"""

# This software was written by Fabian Parzer.
# Copyright owned by the University of Vienna, 2020. All rights reserved.

from math import sqrt
import numpy as np

from regop import RegularizationOperator, DiagonalOperator, ScaledOperator

from .constraints import Constraint
from .utils import is_vector


class Parameter:
    """
    Container class for a user-defined parameter.
    The prior is determined via a _mean, regop, and the regularization.
    :attr mean: The prior mean.
    :attr regop: A (non-necessarily symmetric) square-root of the prior covariance matrix.
    :attr n: The dimension of the parameter.
    :attr r: The dimension of the domain of 'regop'. Must be larger than or equal to n.
    """
    def __init__(self, dim, mean, regop: RegularizationOperator, eqcon: Constraint, incon: Constraint, lb,
                 options):
        self.dim = dim
        self.mean = mean
        self.regop = regop
        self.rdim = regop.rdim
        self.eqcon = eqcon
        self.incon = incon
        self.lb = lb
        self.options = options


class Parameters(list):
    """
    A child class of 'list'.
    You should add parameters with 'addParameter'. Using Parameters' 'append' method
    is strongl advised against.
    :attr nparams: The number of parameters.
    :attr _mean: The concatenated _mean of all parameters.
    :attr regop_list: A list of all covariance operators corresponding to the parameters.
    """
    def __init__(self):
        list.__init__(self)
        self.nparams = 0
        self.dim = 0    # overall dimension of all parameters combined
        self.rdim = 0    # overall r of all parameters combined
        self.mean = None
        self.regop_list = []

    def copy(self):
        """
        Returns a copy of itself.
        :return: Parameters
        """
        param_copy = Parameters()
        for param in self:
            param_copy._add_param(param)
        return param_copy

    def addParameter(self, dim, mean=None, alpha=None, regop=None, eq=None, ineq=None, lb=None, options=None,
                     verbose=True):
        """
        Adds a new parameter with regularization term ||_covroot^(-1)(x-_mean)||_reg
        :param mean: The prior _mean for the parameter. It also determines the parameter dimension and must be given.
        :param regop: A square-root of the prior covariance. Must be an Covroot object.
        :param options: Further options for future regularization types. Not yet implemented.
        :param verbose: A Boolean variable. If True, the dimension and regularization type of the newly added
        parameter are displayed. Good for logging.
        :return: void
        """
        self._check_input(dim, mean, alpha, regop, options)
        if mean is None:
            mean = np.zeros(dim)
        if alpha is None:
            alpha = 1.
        if regop is None:
            param_regop = DiagonalOperator(dim=dim, s=sqrt(alpha))
        else:
            param_regop = ScaledOperator(alpha=alpha, p=regop)
        # create constraints
        if eq is None: eq = {}
        if ineq is None: ineq = {}
        eqcon = Constraint(dim=dim, mat=eq.setdefault("mat", None), vec=eq.setdefault("vec", None))
        incon = Constraint(dim=dim, mat=ineq.setdefault("mat", None), vec=ineq.setdefault("vec", None))
        newparam = Parameter(dim, mean, param_regop, eqcon, incon, lb, options)
        self._add_param(newparam)
        if verbose:
            print(f"New parameter of dimension {newparam.dim} added.")
            print("Total number of parameters: ", self.nparams)

    def _add_param(self, param: Parameter):
        self.append(param)
        if self.mean is None:
            self.mean = param.mean
        else:
            self.mean = np.concatenate((self.mean, param.mean), axis=0)
        self.nparams += 1
        self.dim += param.dim
        self.rdim += param.rdim

    def _check_input(self, dim, mean, alpha, regop, options):
        if mean is not None:
            assert is_vector(mean)
            assert mean.size == dim
        if alpha is not None:
            assert alpha > 0
        if options is not None:
            assert isinstance(options, dict)
        if regop is not None:
            assert regop.dim == dim

    def extract_x(self, x):
        # given concatenated vector x, extracts tuple of parameters
        assert x.size == self.dim
        pvalues = []
        d0 = 0
        for i in range(self.nparams):
            param = self[i]
            dpa = param.dim
            pvalues.append(x[d0:d0+dpa])
            d0 += dpa
        return pvalues

    def extract_w(self, w):
        # given concatenated vector w, extracts w_1, ..., w_l
        assert w.size == self.rdim
        wlist = []
        r = 0
        for i in range(self.nparams):
            param = self[i]
            r_i = param.r
            wlist.append(w[r:r+r_i])
            r += r_i
        return wlist

