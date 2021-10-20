
import numpy as np

from cgn.cnls_solve.cgn_state import CGNState
from cgn import IdentityOperator


def test_cgn():
    m = 50
    n = 10
    def jacfun(x):
        return np.random.randn(m, n)
    cgn_state = CGNState(jac=jacfun, q=IdentityOperator(dim=n))
    cgn_state.x = np.ones(n)
    j1 = cgn_state.jac
    j2 = cgn_state.jac
    # j1 and j2 must be equal, even though jacfun always return something different
    # because Jacobian is only computed once.
    assert np.isclose(j1, j2).all()