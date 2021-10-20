"""
Tests ggn_new.solvers.GaussNewton on an unconstrained linear least-squares problem.
"""

# This software was written by Fabian Parzer.
# Copyright owned by the University of Vienna, 2020. All rights reserved.

import numpy as np
import cgn

from tests.acceptance.test_linear import LinearProblem
from tests.acceptance.do_test import do_test


class LinearConstrainedProblem(LinearProblem):

    def __init__(self):
        LinearProblem.__init__(self)
        # add sum_to_one constraint
        n = self._problem.n
        a = np.ones((1, n))
        b = a @ self._minimizer
        # make sure that initial guess satisfies constraint
        x0 = np.ones(n)
        x0 = x0 / np.sum(x0) * b
        assert np.isclose(a @ x0, b).all()
        self._problem.add_equality_constraint(a, b)


def test_linear_constrained():
    lcp = LinearConstrainedProblem()
    do_test(lcp)
