"""
Contains class "State"
"""

class State:
    """
    Manages the state of the bounded Gauss-Newton method.
    :attr s: the current position
    :attr w: w = Ps - s_tilde
    :attr f: the function
    """
    def __init__(self, jac):
        self.x = None
        self.w = None
        self.f = None
        self._jac = jac
        self._j = None

    def j(self):
        """
        If called for the first time, it computes the Jacobian at State.x.
        If called a second time, it uses the stored value.
        :return: ndarray
            The Jacobian at State.x
        """
        if self._j is None:
            self._j = self._jac(self.x)
        return self._j