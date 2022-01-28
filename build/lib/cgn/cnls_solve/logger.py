"""
Contains class "Logger"
"""

import logging
import numpy as np
from prettytable import PrettyTable

from ..cnls_solve.cnls_solution import CNLSSolution, OptimizationStatus

COLUMN_NAMES = ["Iteration",
                "Cost                   ",
                "Constraint violation   ",
                "Stepsize (||p||)       ",
                "Steplength (h)         ",
                "Computation time [s]   "]


class Logger:
    def __init__(self, verbosity, filename):
        """
        :param verbosity: int
            Level of verbosity:
            - 0: no output
            - 1: output to logfile
            - 2: output to console at running time
            - 3: output both to console and logfile
        :param filename: str, optional
            Specifies name of logfile. Defaults to "cgn.log".
        """
        self.verbosity = verbosity
        self.filename = filename
        self._logger = logging.getLogger()
        if verbosity == 1 or verbosity == 3:
            file_log_handler = logging.FileHandler(filename)
            self._logger.addHandler(file_log_handler)
        if verbosity == 2 or verbosity == 3:
            out_log_handler = logging.StreamHandler()
            self._logger.addHandler(out_log_handler)
        self._logger.setLevel(20)
        self._table = PrettyTable(COLUMN_NAMES)

    def print_preamble(self, cost_start: float):
        self._logger.info("\n")
        self._logger.info(f"Starting the constrained Gauss-Newton method. Cost at starting value: {cost_start}")

    def print_column_names(self):
        self._logger.info(self._table)

    def print_iteration_info(self, k, cost, cviol, p, steplength, time):
        stepsize = np.linalg.norm(p)
        self._table.add_row([k, cost, cviol, stepsize, steplength, time])
        self._logger.info("\n".join(self._table.get_string().splitlines()[-2:]))

    def print_epilogue(self, solution: CNLSSolution):
        niter = solution.niter
        status = solution.status
        if status == OptimizationStatus.converged:
            self._logger.info(f"The iteration converged successfully after {niter} steps.")
            self._logger.info(f"Minimum of cost function: {solution.min_cost}")
        elif status == OptimizationStatus.maxout:
            self._logger.info(f"The iteration stopped as it reached the maximum number of iterations.")
        elif status == OptimizationStatus.timeout:
            self._logger.info(f"The iteration stopped due to timeout.")
        elif status == OptimizationStatus.constraint_violated:
            self._logger.info(f"Warning: The optimization was not able to satisfy the constraints within the"
                              f" given tolerance.")
        else:
            self._logger.info(f"The iteration stopped due to an unknown error.")
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)
