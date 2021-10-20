
import numpy as np

from cgn.cnls_solve.optimization_solution import OptimizationSolution

def test_optimization_solution():
    n = 10
    minimizer = [np.zeros(n)]
    precision = np.ones((n, n))
    def cost_function(x):
        return np.sum(np.square(x))
    min_cost = cost_function(*minimizer)
    success = True
    info = {"niter": 5}
    optsol = OptimizationSolution(minimizer, precision, min_cost, success, info)
    # change minimizer, optsol should be frozen
    minimizer = [np.ones(n)]
    assert np.all(optsol.minimizer[0] == np.zeros(n))


