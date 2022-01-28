
import numpy as np
import pytest

from cgn import LinearConstraint, Parameter


def test_linear_constraint():
    n = 13
    c = 3
    # create parameter
    x = Parameter(dim=n, name="x")
    a = np.random.randn(c, n)
    b = np.random.randn(c)
    eqcon = LinearConstraint(parameters=[x], a=a, b=b, ctype="eq")
    assert eqcon.dim == n
    assert eqcon.cdim == c
    assert eqcon.ctype == "eq"
    assert np.isclose(eqcon.a, a).all()
    assert np.isclose(eqcon.b, b).all()
    # Check that we cannot change attributes
    with pytest.raises(Exception) as e:
        eqcon.a = np.random.randn(c, n)
        eqcon.b = np.random.randn(c)
        eqcon.ctype = "foo"
    # Check that dimensions must match
    y = Parameter(dim=n+2, name="y")
    with pytest.raises(Exception) as e:
        newcon = LinearConstraint(parameters=[y], a=a, b=b, ctype="ineq")
    # Check that ctype is checked
    with pytest.raises(Exception) as e:
        con2 = LinearConstraint(parameters=[x], a=a, b=b, ctype="wrong")


def test_multiple_parameters():
    # Check that we can also define constraints for multiple parameters at once.
    n1 = 13
    n2 = 1
    c = 1
    x1 = Parameter(dim=n1, name="x1")
    x2 = Parameter(dim=n2, name="x2")
    a = np.random.randn(c, n1+n2)
    b = np.random.randn(c)
    eqcon = LinearConstraint(parameters=[x1, x2], a=a, b=b, ctype="eq")
    assert eqcon.dim == n1 + n2
    assert eqcon.cdim == c
    assert eqcon.ctype == "eq"
    assert np.isclose(eqcon.a, a).all()
    assert np.isclose(eqcon.b, b).all()