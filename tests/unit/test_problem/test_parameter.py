
import numpy as np
import pytest

from cgn import Parameter
from cgn.regop import IdentityOperator, MatrixOperator


def test_parameter():
    d1 = 10
    d2 = 1
    x = Parameter(dim=d1, name="x")
    # Check defaults
    assert x.dim == d1
    assert x.beta == 0.
    assert np.isclose(x.mean, np.zeros(d1)).all()
    assert isinstance(x.regop, IdentityOperator)
    assert x.name == "x"
    beta = 42.
    x.beta = beta
    assert x.beta == beta
    m = np.arange(d1)
    x.mean = m
    assert np.isclose(x.mean, m).all()
    r1 = d1 - 1
    r = np.random.randn(r1, d1)
    x.regop = MatrixOperator(mat=r)
    assert np.isclose(x.regop.mat, r).all()
    assert x.rdim == r1
    y = Parameter(dim=d2, name="y")
    assert np.isclose(y.mean, np.zeros((d2,))).all()


def test_parameter_exceptions():
    d = 10
    d_wrong = 7
    x = Parameter(dim=d, name="x")
    with pytest.raises(Exception) as e1:
        x.beta = np.arange(d)   # cannot assign vector to beta
        print(x.beta)
    with pytest.raises(Exception) as e2:
        x.mean = np.arange(d_wrong)     # mean must have correct dimension
    with pytest.raises(Exception) as e3:
        x.regop = np.random.randn(d, d)     # cannot assign matrix to regularization operator
    with pytest.raises(Exception) as e4:
        r = IdentityOperator(dim=d_wrong)
        x.regop = r     # regularization operator must have correct dimension.


def test_bound():
    d = 10
    d_wrong = 7
    lb1 = np.zeros(d)
    lb2 = np.zeros(d_wrong)
    x = Parameter(dim=d, name="x")
    # check that default is no lower bound
    assert np.all(x.lb <= - np.inf)
    x.lb = lb1
    assert np.isclose(x.lb, lb1).all()
    with pytest.raises(Exception) as e:
        x.lb = lb2  # lower bound must have matching dimension


def test_rdim():
    d = 10
    r = 20
    x = Parameter(dim=d, name="x")
    regop = np.random.randn(r, d)
    x.regop = MatrixOperator(regop)
    assert x.rdim == r

