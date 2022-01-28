
import numpy as np
import pytest

from cgn.translator.translated_solution import TranslatedSolution
from cgn import Parameter


def test_translated_solution():
    n1 = 10
    n2 = 3
    n = n1 + n2
    x = Parameter(dim=n1, name="x")
    y = Parameter(dim=n2, name="y")
    p = np.random.randn(n, n)
    cost = 4.2
    success = False
    niter = 42
    xmin = np.random.randn(n1)
    ymin = np.random.randn(n2)
    solution = TranslatedSolution(parameters=[x, y], minimizers=[xmin, ymin], precision=p, cost=cost, success=success,
                                  niter=niter)
    xsol = solution.minimizer("x")
    ysol = solution.minimizer("y")
    assert np.isclose(xsol, xmin).all()
    assert np.isclose(ysol, ymin).all()
    assert np.isclose(solution.precision, p).all()
    with pytest.raises(Exception) as e:
        zsol = solution.minimizer("z")