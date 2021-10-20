"""
Contains classes "Translator".
"""

import numpy as np

from ..cnls_solve import CNLS
from ..cnls_solve.optimization_solution import OptimizationSolution, OptimizationStatus
from ..regop import scale_operator, make_block_operator

from .problem import Problem
from .problem_solution import ProblemSolution


class Translator:
    """
    Translates a cgn.Problem object to a CNLS object.
    """
    def __init__(self, problem: Problem):
        self._problem = problem
        self._nparams = problem.parameters.nparams

    def translate(self) -> CNLS:
        """
        Returns a CNLS equivalent to the MultiParameterProblem
        :return: CNLS
        """
        fun = self._modify_function(self._problem.fun)
        jac = self._modify_function(self._problem.jac)
        q = self._problem.q
        eqcon = self._problem.equality_constraint
        incon = self._problem.inequality_constraints
        mean = self._combine_means()
        r = self._combine_regops()
        lb = self._problem.lower_bound
        scale = self._problem.scale
        cnls = CNLS(func=fun, jac=jac, q=q, r=r, m=mean, eqcon=eqcon, incon=incon, lb=lb, scale=scale)
        return cnls

    def _modify_function(self, func):
        """
        Takes function that takes list of arguments and transforms it to function that takes concatenated
        vector as input.
        :param func: function that takes a tuple as argument
        :return: function that takes a single vector as argument
        """
        def newfunc(x):
            x_tuple = self._problem.parameters.extract_x(x)
            return func(*x_tuple)
        return newfunc

    def _combine_means(self):
        mean_list = self._problem.parameters.means
        mean = np.concatenate(mean_list)
        return mean

    def combine_x(self, x_list):
        assert len(x_list) == self._nparams
        return np.concatenate(x_list)

    def _combine_regops(self):
        regops_list = []
        n_list = []
        r_list = []
        for param in self._problem.parameters.list:
            scaled_regop = scale_operator(regop=param.regop, alpha=param.beta)
            regops_list.append(scaled_regop)
            n_list.append(param.dim)
            r_list.append(param.rdim)
        op = make_block_operator(operator_list=regops_list)
        return op

    def translate_solution(self, cnls_solution: OptimizationSolution) -> ProblemSolution:
        """
        :param cnls_solution: An object of type CnllsSolution
        :return : An object of type LpcnllsSolution
        """
        xmin = self._problem.parameters.extract_x(cnls_solution.minimizer)
        precision = cnls_solution.precision
        cost = cnls_solution.min_cost
        niter = cnls_solution.niter
        success = (cnls_solution.status == OptimizationStatus.converged)
        multi_parameter_solution = ProblemSolution(minimizer=xmin,
                                                   precision=precision,
                                                   cost=cost,
                                                   success=success,
                                                   niter=niter)
        return multi_parameter_solution
