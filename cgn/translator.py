"""
Contains classes ArgumentTransform and Translator.
These classes serve to transform a LpCNLLS object into
and equivalent CNLLS object.
"""

import numpy as np

from regop import BlockOperator

from .constraints import combine_constraints
from .multi_parameter_problem import MultiParameterProblem, MultiParameterSolution
from .cnls import CNLS, CNLSSolution


class Translator:
    """
    Translates a MultiParameterProblem into a BNLS problem to which
    the iterations can then be applied.
    """
    def __init__(self, problem: MultiParameterProblem):
        self._mp = problem
        self._nparams = problem.params.nparams

    def translate(self) -> CNLS:
        """
        Returns a CNLS equivalent to the MultiParameterProblem
        :return: CNLS
        """
        func = self._modify_function(self._mp.fun)
        jac = self._modify_function(self._mp.jac)
        eqcon, incon = self._combine_equality_inequality_constraints()
        p = self._combine_regops()
        xbar = self._combine_xbar()
        lb = self._combine_bounds()
        cnls = CNLS(func=func, jac=jac, regop=p, xbar=xbar, eqcon=eqcon, incon=incon, lb=lb)
        return cnls

    def _modify_function(self, func):
        """
        Takes function that takes list of arguments and transforms it to function that takes concatenated
        vector as input.
        :param func: function that takes a tuple as argument
        :return: function that takes a single vector as argument
        """
        def newfunc(x):
            x_tuple = self._mp.params.extract_x(x)
            return func(*x_tuple)
        return newfunc

    def _combine_equality_inequality_constraints(self):
        """
        Given the multiple parameters, each with an equality and an inequality constraint,
        create an equality and an inequality constraint for the concatenated parameter.
        :return: EqualityConstraint, InequalityConstraint
        """
        # build list of constraints
        eqcon_list = []
        incon_list = []
        for param in self._mp.params:
            eqcon_list.append(param.eqcon)
            incon_list.append(param.incon)
        # then, combine all constraints in the list into a single constraint
        eqcon_combined = combine_constraints(eqcon_list)
        incon_combined = combine_constraints(incon_list)
        return eqcon_combined, incon_combined

    def _combine_bounds(self):
        lb_list = []
        all_none = True
        for param in self._mp.params:
            if param.lb is None:
                lb = - np.inf * np.ones(param.dim)
            else:
                all_none = False
                lb = param.lb
            lb_list.append(lb)
        # if there are no lower bounds, return None
        if all_none:
            return None
        else:
            return np.concatenate(lb_list)

    def _combine_xbar(self):
        xbar_list = []
        for param in self._mp.params:
            xbar_list.append(param.mean)
        mean = np.concatenate(xbar_list)
        return mean

    def combine_x(self, x_list):
        assert len(x_list) == self._nparams
        return np.concatenate(x_list)

    def _combine_regops(self):
        regops_list = []
        n_list = []
        r_list = []
        for param in self._mp.params:
            regops_list.append(param.regop)
            n_list.append(param.dim)
            r_list.append(param.rdim)
        op = BlockOperator(operator_list=regops_list, dim_list=n_list, rdim_list=r_list)
        return op

    def translate_solution(self, cnls_solution: CNLSSolution) -> MultiParameterSolution:
        """
        :param cnls_solution: An object of type CnllsSolution
        :return : An object of type LpcnllsSolution
        """
        xmin = self._mp.params.extract_x(cnls_solution.minimizer)
        precision = cnls_solution.precision
        cost = cnls_solution.cost
        costfun = self._translate_costfun(cnls_solution.costfun)
        # if x_map is a list of one element, directly return x_map
        if len(xmin) == 1:
            xmin = xmin[0]
        info = cnls_solution.info
        multi_parameter_solution = MultiParameterSolution(xmin, precision, cost, costfun, info)
        return multi_parameter_solution

    @staticmethod
    def _translate_costfun(costfun):
        """
        Translate a BNLS-costfunction (i.e. one that depends on one big vector) to a multiparameter-costfunction
        (i.e. one that depends on multiple vectors)
        :param costfun: function
            The costfunction return by a BNLSSolution
        :return: function
            The new costfunction that takes a tuple as argument.
        """
        def new_costfun(*x_tuple):
            x = np.concatenate(x_tuple)
            return costfun(x)
        return new_costfun
