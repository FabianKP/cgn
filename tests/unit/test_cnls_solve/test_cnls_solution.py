
import numpy as np

from cgn.cnls_solve.cnls_solution import CNLSSolution, OptimizationStatus


def test_cnls_solution():
    n = 10
    minimizer = np.zeros(n)
    precision = np.ones((n, n))
    def cost_function(x):
        return np.sum(np.square(x))
    min_cost = cost_function(minimizer)
    success = OptimizationStatus.converged
    optsol = CNLSSolution(minimizer, precision, min_cost, success, niter=5)
    # change minimizer, optsol should be frozen
    minimizer = np.ones(n)
    assert np.all(optsol.minimizer[0] == np.zeros(n))


