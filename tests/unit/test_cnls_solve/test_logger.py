
import numpy as np
import time

from cgn.cnls_solve.logger import Logger
from cgn.cnls_solve.cnls_solution import CNLSSolution, OptimizationStatus


def test_logger_instance():
    logger = Logger(verbosity=3, filename="test.log")
    cost_start = "[COST AT START]"
    logger.print_preamble(cost_start)
    maxiter = 13
    n = 10
    cviol = 1e-12
    t0 = time.time()
    logger.print_column_names()
    for k in range(maxiter):
        cost = np.random.randn(1)[0]
        p = np.random.randn(n)
        h = 0.5 ** k
        t1 = time.time() - t0
        logger.print_iteration_info(k=k, cost=cost, p=p, cviol=cviol, steplength=h, time=t1)
    dummy_solution = CNLSSolution(minimizer="[MINIMIZER]",
                                          min_cost="[MINIMUM]",
                                          precision=None,
                                          status=OptimizationStatus.converged,
                                          niter=maxiter)
    logger.print_epilogue(dummy_solution)

def test_logger():
    test_logger_instance()
    test_logger_instance()
