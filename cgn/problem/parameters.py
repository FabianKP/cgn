"""
Contains class "Parameters".
"""

from copy import deepcopy
from numpy.typing import ArrayLike
from typing import List

from ..regop import RegularizationOperator
from .parameter import Parameter


class Parameters:
    """
    """
    def __init__(self, dims: List[int]):
        self._nparams = len(dims)
        self._dims = deepcopy(dims)
        # initialize with unregularized parameters
        self._parameter_list = []
        self._positions = []
        self._dim = 0    # overall dimension of all parameters combined
        self._rdim = 0    # overall r of all parameters combined
        self._mean_list = []
        self._regop_list = []
        for i in range(self._nparams):
            pdim = dims[i]
            param = Parameter(dim=pdim)
            self._parameter_list.append(param)
            self._positions.append(self._dim)
            self._dim += pdim
            self._mean_list.append(param.mean)
            self._regop_list.append(param.regop)

    @property
    def dim(self):
        return deepcopy(self._dim)

    @property
    def list(self) -> List[Parameter]:
        """
        Returns a list of all parameter objects
        :return: List[Parameter]
        """
        return deepcopy(self._parameter_list)

    @property
    def means(self):
        """
        Returns the list of all means.
        :return: List[ArrayLike]
        """
        return deepcopy(self._mean_list)

    @property
    def nparams(self):
        return self._nparams

    @property
    def rdim(self):
        return deepcopy(self._rdim)

    @property
    def regops(self):
        """
        Returns list of all regularization operators.
        :return: list[cgn.RegularizationOperator]
        """
        return deepcopy(self._regop_list)

    def change_parameter(self, paramno, mean: ArrayLike = None, regop: RegularizationOperator = None, beta: float = 1.):
        """
        Changes a parameter of given number.
        """
        # create new parameter
        paramdim = self._parameter_list[paramno].dim
        newparam = Parameter(dim=paramdim, mean=mean, regop=regop, beta=beta)
        # replace old parameter with new parameter
        self._parameter_list[paramno] = newparam
        # adapt other attributes
        self._mean_list[paramno] = newparam.mean
        self._regop_list[paramno] = newparam.regop

    def extract_x(self, x):
        # given concatenated vector x, extracts tuple of parameters
        assert x.size == self.dim
        pvalues = []
        d0 = 0
        for i in range(self._nparams):
            param = self._parameter_list[i]
            dpa = param.dim
            pvalues.append(x[d0:d0+dpa])
            d0 += dpa
        return pvalues

    def extract_w(self, w):
        # given concatenated vector w, extracts w_1, ..., w_l
        assert w.size == self.rdim
        wlist = []
        r = 0
        for i in range(self._nparams):
            param = self._parameter_list[i]
            r_i = param.rdim
            wlist.append(w[r:r+r_i])
            r += r_i
        return wlist

    def position(self, i):
        return self._positions[i]

    # PROTECTED

    @staticmethod
    def _check_input(dim, mean, alpha, regop, options):
        if mean is not None:
            assert mean.shape == (dim, )
        if alpha is not None:
            assert alpha > 0
        if options is not None:
            assert isinstance(options, dict)
        if regop is not None:
            assert regop.dim == dim
