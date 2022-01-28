
import numpy as np
import pytest

from cgn import LinearConstraint, Parameter, Problem, IdentityOperator



@pytest.fixture
def one_dim_problem():
    x = Parameter(dim=1, name="x")
    x.lb = np.zeros((1,))
    def fun(x) -> np.ndarray:
        return x ** 2
    def jac(x) -> np.ndarray:
        return 2 * x
    problem = Problem(parameters=[x], fun=fun, jac=jac)
    return problem


@pytest.fixture
def three_parameter_problem():
    n1 = 20
    n2 = 50
    n3 = 1
    x = Parameter(dim=n1, name="x")
    y = Parameter(dim=n2, name="y")
    z = Parameter(dim=n3, name="z")
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
    def fun(x, y, z):
        return np.square(np.concatenate((x, y, z), axis=0))

    def jac(x, y, z):
        return 2 * np.diagflat(np.concatenate((x, y, z), axis=0))
    problem = Problem(parameters=[x, y, z], fun=fun, jac=jac, scale=scale, constraints=[eqcon, incon])
    return problem


@pytest.fixture
def unconstrained_problem():
    n1 = 20
    n2 = 3
    x = Parameter(dim=n1, name="x")
    y = Parameter(dim=n2, name="y")
    x.beta = 3
    y.mean = np.arange(n2)
    def fun(x, y):
        return np.square(np.concatenate((x, y), axis=0))
    def jac(x, y):
        return 2 * np.diagflat(np.concatenate((x, y), axis=0))
    problem = Problem(parameters=[x, y], fun=fun, jac=jac)
    return problem