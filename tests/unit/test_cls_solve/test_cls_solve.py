
import numpy as np

from cgn.cls_solve import cls_solve, CLS


def test_unconstrained():
    n = 100
    h = np.eye(100)
    x = np.arange(n)
    y = h @ x
    cls = CLS(h=h, y=y)
    x_min = cls_solve(cls)
    assert np.isclose(x_min, x).all()


def test_constrained():
    n = 1000
    # create tests problem
    h = np.eye(n)
    x = np.arange(n)
    y = h @ x
    a = np.ones((1, n))
    b = np.sum(x) * np.ones((1, ))
    l = np.zeros(n)
    l[-1] = -np.inf
    c = np.ones((1, n))
    d = b - 1.
    cls = CLS(h=h, y=y, a=a, b=b, c=c, d=d, l=l)
    x_min = cls_solve(cls)
    # check that the minimizer is good
    assert np.isclose(x_min, x).all()
