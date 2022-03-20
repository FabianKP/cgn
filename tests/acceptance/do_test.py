
import numpy as np

import cgn
from tests.acceptance.problem import TestProblem


def do_test(test: TestProblem):
    # Initialize solver.
    solver = cgn.CGN()
    # Set solver options equal to the problem-specific options
    solver.options = test.options
    # Solve problem with cgn:
    solution = solver.solve(problem=test.cgnproblem)
    # Compute the cost of the cgn-minimizer.
    cost = solution.cost
    minimizer = solution.minimizer_tuple
    cost_min = test.cgnproblem.costfun(*minimizer)
    assert np.isclose(cost, cost_min)
    # Compare the cost to the reference value
    assert cost <= (1 + test.tol) * test.minimum