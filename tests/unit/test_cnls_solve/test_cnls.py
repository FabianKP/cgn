import numpy as np

from cgn.regop import IdentityOperator
from cgn.cnls_solve.cnls import CNLS
from cgn.cnls_solve import ConcreteConstraint

def test_cnls():
    m = 50
    n = 20
    h = np.random.randn(m, n)
    x = np.random.randn(n)
    mean = np.zeros(n)
    regop = IdentityOperator(dim=n)
    def fun(x):
        return h @ x
    def jac(x):
        return h
    lb = x.copy()
    a = np.ones((1, n))
    b = np.sum(x) * np.ones((1, ))
    c = a.copy()
    d = b - 1
    # make CNLS
    def eqfun(x):
        return a @ x - b
    def eqjac(x):
        return a
    def infun(x):
        return c @ x - d
    def injac(x):
        return c
    eqcon = ConcreteConstraint(dim=n, fun=eqfun, jac=injac)
    incon = ConcreteConstraint(dim=n, fun=infun, jac=injac)
    ub = np.inf * np.ones((n, ))
    cnls = CNLS(func=fun, jac=jac, q=IdentityOperator(dim=n), m=mean, r=regop, eqcon=eqcon, incon=incon, lb=lb, ub=ub,
                scale=1.)
    # first, check that x indeed satisfies the constraints.
    assert cnls.satisfies_constraints(x)
    # mean does not satisfy the constraint
    assert not cnls.satisfies_constraints(mean)