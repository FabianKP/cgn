
import numpy as np
from typing import Sequence

from ..problem import Parameter, ProblemSolution


class TranslatedSolution(ProblemSolution):

    def __init__(self, parameters: Sequence[Parameter], minimizers: Sequence[np.ndarray], precision: np.ndarray,
                 cost: float, success: bool, niter: int):
        self._check_input(parameters, minimizers, precision, cost, success)
        self._parameters = parameters
        self._minimizer_tuple = minimizers
        self._precision = precision
        self._cost = cost
        self._success = success
        self._niter = niter

    def minimizer(self, pname: str) -> np.ndarray:
        """
        :raises: Exception, if no parameter with name ``pname`` is found.
        """
        # Find the number of the parameter or throw an error.
        try:
            pnumber = self._find_pnumber(pname)
        except:
            raise Exception(f"No parameter found with name {pname}.")
        # Get the corresponding minimizer
        pmin = self._minimizer_tuple[pnumber]
        return pmin

    def _check_input(self, parameters: Sequence[Parameter], minimizers: Sequence[np.ndarray], precision: np.ndarray,
                     cost: float, success: bool):
        assert len(parameters) == len(minimizers)
        for minimizer, parameter in zip(minimizers, parameters):
            assert minimizer.shape == (parameter.dim, )
        assert cost >= 0
        overall_dim = 0
        for param in parameters:
            overall_dim += param.dim
        assert precision.shape == (overall_dim, overall_dim)
        assert success in [True, False]

    def _find_pnumber(self, pname: str):
        """
        Finds the number of the parameter with name "pname" in self._parameters.

        :raises: Exception, if no parameter with the corresponding name is found.
        """
        found = False
        for i in range(len(self._parameters)):
            param_i = self._parameters[i]
            if param_i.name == pname:
                return i
        if not found:
            raise Exception





