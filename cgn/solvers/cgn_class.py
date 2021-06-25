"""
Contains class "BGN"
"""

# This software was written by Fabian Parzer.
# Copyright owned by the University of Vienna, 2020. All rights reserved.


from .constrained_gauss_newton import ConstrainedGaussNewton
from ..utils import is_vector
from ..translator import Translator
from ..multi_parameter_problem import MultiParameterProblem, MultiParameterSolution


class CGN:
    """
    Interface for the bounded Gauss-Newton method.
    """
    def __init__(self, problem: MultiParameterProblem):
        """
        :param problem: An object of type BNLS.
        """
        self._params = problem.params
        self._translator = Translator(problem)
        cnls = self._translator.translate()
        self._method = ConstrainedGaussNewton(cnls)
        self._starting_values = self._find_starting_value()

    def solve(self, options=None) -> MultiParameterSolution:
        """
        Computes the solution of the bounded nonlinear least-squares problem.
        :param options: dict
            Contains runtime options for the method. See the documentation
            of the corresponding method and Linesearch for the used options.
        :return: MultiParameterSolution
        """
        if options is None:
            myoptions = None
        else:
            myoptions = options.copy()
        x_start = self._translator.combine_x(self._starting_values)
        cnls_solution = self._method.iterate(x_start, myoptions)
        solution = self._translator.translate_solution(cnls_solution)
        return solution

    def set_starting_value(self, pnumber, val):
        """
        Set a user-defined starting value for the _solver.
        :param pnumber: int
            Number of the parameter for which the starting value should be set.
        :param val: ndarray
            starting value. Must have same shape as the parameter _mean.
        :exception if the dimension of the starting value doesn't equal the dimension of the parameter _mean.
        """
        assert is_vector(val)
        assert len(self._starting_values[pnumber]) == len(val), "Cannot change the dimension of an existing parameter."
        try:
            self._starting_values[pnumber] = val
        except:
            raise Exception()

    def _find_starting_value(self):
        """
        Chooses default starting values for the parameters.
        :return: list
            a list of the starting values corresponding to each parameter.
        """
        starting_values = []
        for param in self._params:
            starting_values.append(param.mean)
        return starting_values
