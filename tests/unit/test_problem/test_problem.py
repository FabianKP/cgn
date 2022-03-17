
import numpy as np
import pytest

from cgn.regop import IdentityOperator
from cgn import LinearConstraint, Parameter, Problem


def test_problem():
    # Initialize parameters
    n1 = 20
    n2 = 50
    x = Parameter(start=np.zeros(n1), name="x")
    y = Parameter(start=np.zeros(n2), name="y")
    x.beta = 0.384
    y.beta = 32.2
    x.lb = np.zeros(n1)
    x.mean = np.random.randn(n1)
    y.mean = np.random.randn(n2)
    x.regop = IdentityOperator(dim=n1)
    y.regop = IdentityOperator(dim=n2)
    # Initialize constraints
    a = np.ones((1, n1 + n2))
    b = np.ones((1,))
    eqcon = LinearConstraint(parameters=[x, y], a=a, b=b, ctype="eq")
    c = np.eye(n2)
    d = np.ones(n2)
    incon = LinearConstraint(parameters=[y], a=c, b=d, ctype="ineq")
    scale = 0.1
    # Initialize misfit function and Jacobian
    def fun(x1, x2):
        return np.square(np.concatenate((x1, x2), axis=0))
    def jac(x1, x2):
        return 2 * np.diagflat(np.concatenate((x1, x2), axis=0))
    problem = Problem(parameters=[x, y], fun=fun, jac=jac, constraints=[eqcon, incon], scale=scale)
    # Check that the correct problem was initialized
    assert isinstance(problem.q, IdentityOperator)
    assert problem.nparams == 2
    assert problem.n == n1 + n2


def test_constraints_must_depend_on_problem_parameters():
    n1 = 20
    n2 = 50
    n3 = 1
    x = Parameter(start=np.zeros(n1), name="x")
    y = Parameter(start=np.zeros(n2), name="y")
    z = Parameter(start=np.zeros(n3), name="z")
    a1 = np.random.randn(n1 + n2 + n3, n1 + n2 + n3)
    b1 = np.random.randn(n1 + n2 + n3)
    con1 = LinearConstraint(parameters=[x, y, z], a=a1, b=b1, ctype="eq")
    def fun(x1, x2):
        return np.square(np.concatenate((x1, x2), axis=0))
    def jac(x1, x2):
        return 2 * np.diagflat(np.concatenate((x1, x2), axis=0))
    with pytest.raises(Exception) as e1:
        prob1 = Problem(parameters=[x, y], fun=fun, jac=jac, constraints=[con1])
    a2 = np.random.randn(n3, n3)
    b2 = np.random.randn(n3)
    con2 = LinearConstraint(parameters=[z], a=a2, b=b2, ctype="ineq")
    with pytest.raises(Exception) as e2:
        prob2 = Problem(parameters=[x, y], fun=fun, jac=jac, constraints=[con2])


def test_modify_problem_after_initialization():
    n1 = 20
    n2 = 50
    x = Parameter(start=np.zeros(n1), name="x")
    y = Parameter(start=np.zeros(n2), name="y")
    def fun(x1, x2):
        return np.square(np.concatenate((x1, x2), axis=0))
    def jac(x1, x2):
        return 2 * np.diagflat(np.concatenate((x1, x2), axis=0))
    problem = Problem(parameters=[x, y], fun=fun, jac=jac)
    # Adapt the regularization
    beta = 42
    problem.parameter("x").beta = beta
    assert problem._parameter_list[0].beta == beta
    # Add constraint
    a = np.ones((1, n1 + n2))
    b = np.ones((1,))
    eqcon = LinearConstraint(parameters=[x, y], a=a, b=b, ctype="eq")
    problem.constraints.append(eqcon)
    # Check that the constraint was really added.
    assert eqcon in problem.constraints


def test_no_duplicate_names_allowed():
    n1 = 10
    n2 = 20
    x1 = Parameter(start=np.zeros(n1), name="x")
    x2 = Parameter(start=np.zeros(n2), name="x")
    def fun(x1, x2):
        return np.square(np.concatenate((x1, x2), axis=0))
    def jac(x1, x2):
        return 2 * np.diagflat(np.concatenate((x1, x2), axis=0))
    with pytest.raises(Exception) as e:
        problem = Problem(parameters=[x1, x2], fun=fun, jac=jac)







