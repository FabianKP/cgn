
import numpy as np
import pytest

from cgn.regop import IdentityOperator
from cgn.cnls_solve import CNLS, cnls_solve
from cgn.problem.linear_constraint import LinearConstraint
from cgn.cnls_solve.solveroptions import Solveroptions
from cgn.cnls_solve.linesearch_options import LinesearchOptions


def test_cnls_solve():
    m = 10
    n = 20
    i_n = np.eye(n)
    h = i_n[:m, :]
    x = np.arange(n)
    mean = np.zeros(n)
    regop = IdentityOperator(dim=n)

    def fun(x):
        return h @ x

    def jac(x):
        return h

    lb = np.zeros(n)
    a = np.ones((1, n))
    b = np.sum(x) * np.ones((1,))
    c = a.copy()
    d = b - 1
    # make CNLS
    eqcon = LinearConstraint(dim=n, mat=a, vec=b)
    incon = LinearConstraint(dim=n, mat=c, vec=d)
    cnls = CNLS(func=fun, jac=jac, q=IdentityOperator(dim=n), m=mean, r=regop, eqcon=eqcon, incon=incon, lb=lb,
                scale=1.)
    # solve
    options = Solveroptions()
    options.set_verbosity(0)
    options.set_verbosity(-1)
    assert options.verbosity == 0
    loptions = LinesearchOptions()
    solution = cnls_solve(cnls=cnls, start=x, options=options, linesearch_options=loptions)
    x_min = solution.minimizer
    # the solution should at least satisfy all constraints
    assert np.isclose(a @ x_min, b).all()
    assert np.all(c @ x_min >= d)
    assert np.all(x_min >= lb)
    # check that error is thrown if start is meaningless
    bad_start = np.ones(3)
    with pytest.raises(ValueError) as e_info:
        bad_solution = cnls_solve(cnls, bad_start, options, loptions)

