
import numpy as np
import pytest

from cgn import Parameter
from cgn.regop import MatrixOperator, DiagonalOperator


@pytest.fixture
def x_parameter():
    n = 12
    beta = 42
    mean = np.arange(n)
    regop = DiagonalOperator(dim=n, s=np.arange(1, n+1)**2)
    x = Parameter(dim=n, name="x")
    x.beta = beta
    x.mean = mean
    x.regop = regop
    return x


@pytest.fixture
def y_parameter():
    n = 3
    beta = 0.
    y = Parameter(dim=n, name="y")
    y.beta = beta
    y.lb = np.zeros(3)
    return y


@pytest.fixture
def z_parameter():
    n = 1
    beta = 12
    mean = np.ones(n)
    rmat = np.ones((n, n))
    z = Parameter(dim=n, name="z")
    z.beta = beta
    z.mean = mean
    z.regop = MatrixOperator(rmat)
    return z


@pytest.fixture
def u_parameter():
    n = 4
    r = np.eye(n)[:2, :]
    regop = MatrixOperator(mat=r)
    u = Parameter(dim=n, name="u")
    u.regop = regop
    u.beta = 1.
    return u