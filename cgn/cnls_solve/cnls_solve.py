"""
Contains function "cnls_solve"
"""

from .constrained_gauss_newton import ConstrainedGaussNewton
from .cnls import CNLS
from .linesearch_options import LinesearchOptions
from .solveroptions import Solveroptions


def cnls_solve(cnls: CNLS, start, options: Solveroptions, linesearch_options: LinesearchOptions):
    """
    Solves an CNLS problem using the constrained Gauss-Newton method.
    """
    if start.size != cnls.dim:
        raise ValueError("Dimensions of 'start' and 'cnls' do not match.")
    # initialize ConstrainedGaussNewton object
    constrained_gauss_newton = ConstrainedGaussNewton(cnls, options, linesearch_options)
    # Solve the NLSI problem using the Constrained Gauss-Newton Method.
    cnls_solution = constrained_gauss_newton.solve(start=start)
    return cnls_solution
