"""
Contains function "cnls_solve"
"""
import numpy as np

from .constrained_gauss_newton import ConstrainedGaussNewton
from .cnls import CNLS
from .cnls_solution import CNLSSolution
from .linesearch_options import LinesearchOptions
from .solveroptions import Solveroptions


def cnls_solve(cnls: CNLS, start: np.ndarray, options: Solveroptions, linesearch_options: LinesearchOptions)\
        -> CNLSSolution:
    """
    Solves an CNLS problem using the constrained Gauss-Newton method.
    """
    if start.size != cnls.dim:
        raise ValueError("Dimensions of 'start' and 'cnls' do not match.")
    # initialize ConstrainedGaussNewton object
    constrained_gauss_newton = ConstrainedGaussNewton(cnls, options, linesearch_options)
    # Solve the CNLS problem using the Constrained Gauss-Newton Method.
    cnls_solution = constrained_gauss_newton.solve(start=start)
    return cnls_solution
