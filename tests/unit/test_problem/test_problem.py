
import numpy as np

from cgn.regop import IdentityOperator
from cgn import Problem


def test_problem():
    n1 = 20
    n2 = 50
    beta1 = 0.384
    beta2 = 32.2
    lb1 = np.zeros(n1)
    a = np.ones((1, n1 + n2))
    b = np.ones((1, ))
    c = np.eye(n2)
    d = np.ones(n2)
    mean1 = np.random.randn(n1)
    mean2 = np.random.randn(n2)
    regop1 = IdentityOperator(dim=n1)
    regop2 = IdentityOperator(dim=n2)
    scale = 0.1
    def fun(x1, x2):
        return np.square(np.concatenate((x1, x2), axis=0))
    def jac(x1, x2):
        return 2 * np.diagflat(np.concatenate((x1, x2), axis=0))
    problem = Problem(dims=[n1, n2], fun=fun, jac=jac, scale=scale)
    assert isinstance(problem.q, IdentityOperator)
    problem.set_regularization(paramno=0, m=mean1, beta=beta1, r=regop1)
    problem.set_regularization(paramno=1, m=mean2, beta=beta2, r=regop2)
    problem.set_lower_bound(i=0, lb=lb1)
    assert problem.lower_bound.size == n1 + n2
    problem.add_equality_constraint(a=a, b=b)
    assert problem.equality_constraint.mat.shape == (1, n1+n2)
    problem.add_inequality_constraint(c=c, d=d, i=1)
    assert problem.inequality_constraints.mat.shape == (n2, n1+n2)




