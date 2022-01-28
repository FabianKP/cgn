"""
Contains class "CGNState"
"""

from ..regop import RegularizationOperator


class CGNState:
    """
    Manages the state of the bounded Gauss-Newton method.
    """
    def __init__(self, jac: callable, q: RegularizationOperator):
        self.x = None
        self.w = None   # w = r(x - m)
        self.h = None   # h = q @ f(x)
        self._q = q
        self._jacfun = jac
        self._jac = None

    @property
    def jac(self):
        """
        If called for the first time, it computes the Jacobian at State.x.
        If called a second time, it uses the stored value.
        :return: ndarray
            The Jacobian at State.x
        """
        if self._jac is None:
            self._jac = self._q.fwd(self._jacfun(self.x))
        return self._jac