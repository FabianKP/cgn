
import numpy as np

from cgn import LinearConstraint, Parameter
from cgn.translator.get_sub_matrix import get_sub_matrix


def test_get_sub_matrix():
    n1 = 13
    n2 = 1
    n3 = 3
    c = 10
    x1 = Parameter(start=np.zeros(n1), name="x1")
    x2 = Parameter(start=np.zeros(n2), name="x2")
    x3 = Parameter(start=np.zeros(n3), name="x3")
    a1 = np.random.randn(c, n1)
    a2 = np.random.randn(c, n2)
    a3 = np.random.randn(c, n3)
    b = np.random.randn(c)
    a = np.concatenate([a1, a2, a3], axis=1)
    constraint = LinearConstraint(parameters=[x1, x2, x3], a=a, b=b, ctype="ineq")
    a1t = get_sub_matrix(a, constraint, 0)
    a2t = get_sub_matrix(a, constraint, 1)
    a3t = get_sub_matrix(a, constraint, 2)
    for a, at in zip([a1, a2, a3], [a1t, a2t, a3t]):
        assert np.isclose(a, at).all()

def test_get_sub_matrix2():
    # Test that get_sub_matrix also works for single parameter
    n = 10
    c = 5
    x = Parameter(start=np.zeros(n), name="x")
    a = np.random.randn(c, n)
    b = np.random.randn(c)
    constraint = LinearConstraint(parameters=[x], a=a, b=b, ctype="eq")
    at = get_sub_matrix(a, constraint, 0)
    assert np.isclose(a, at).all()
