
import numpy as np
from typing import List

from .problem import Problem, ProblemSolution, Translator
from .cnls_solve import cnls_solve
from .cnls_solve.linesearch_options import LinesearchOptions
from .cnls_solve.solveroptions import Solveroptions


class CGN:
    """
    Constrained Gauss-Newton solver for nonlinear least-squares problems with linear constraints.

    :ivar options: Allows to modify the options for the solver. See
        :py:class:`cgn.cnls_solve.solveroptions.Solveroptions` for the modifiable options.
    :ivar linesearch: Allows to set options for the linesearch method.
        See :py:class:`cgn.cnls_solve.linesearch_options.LinesearchOptions` for the modifiable options.
    """
    def __init__(self):
        self.options = Solveroptions()
        self.linesearch = LinesearchOptions()

    def solve(self, problem: Problem, starting_values: List[np.ndarray]) -> ProblemSolution:
        """
        Computes the solution of the constrained nonlinear least-squares problem using the
        constrained Gauss-Newton method.

        :param problem: The optimization problem to solve.
        :param starting_values: The starting values for the optimization.
            The length of the list must be equal to :py:attr:`Problem.nparams`.
        :return: The solution to the optimization problem.
        """
        translator = Translator(problem)
        # Translate the multi-parameter problem to a CNLS problem.
        cnls = translator.translate()
        # Translate the starting value for multi-parameter problem to starting value for cnls problem
        x_start = translator.combine_x(starting_values)
        # Check that the starting value satisfies the constraints.
        if not cnls.satisfies_constraints(x_start, tol=self.options.ctol):
            raise ValueError(f"The starting value must satisfy the constraints with accuracy {self.options.ctol}")
        # Then, solve the CNLS problem using "cnls_solve".
        cnls_solution = cnls_solve(cnls=cnls, start=x_start, options=self.options, linesearch_options=self.linesearch)
        # Translate the solution of the CNLS problem to a solution of the orginal multi parameter problem.
        solution = translator.translate_solution(cnls_solution)
        # Finally, this solution is returned.
        return solution
