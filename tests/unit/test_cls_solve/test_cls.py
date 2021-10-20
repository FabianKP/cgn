
import numpy as np
import pytest

from cgn.cls_solve.cls import CLS


def test_inconsistent_input():
    h = np.eye(5)
    y_bad = np.ones(3)
    with pytest.raises(AssertionError) as e_info:
        cls = CLS(h=h, y=y_bad)
    y = np.ones(5)
    a = np.ones((4, 5))
    b_bad = np.ones(3)
    with pytest.raises(AssertionError) as e_info:
        cls = CLS(h=h, y=y, a=a, b=b_bad)

def test_consistent_input():
    h = np.eye(5)
    y = np.ones(5)
    c = np.ones((1, 5))
    d = np.ones((1,))
    cls = CLS(h=h, y=y, c=c, d=d)
    # cls.l should be - inf
    assert np.all(cls.l <= -np.inf)
    assert cls.a is None
    assert cls.b is None
    lb = np.zeros(5)
    cls = CLS(h=h, y=y, c=c, d=d, l=lb)
    assert np.isclose(lb, cls.l).all()