"""
Contains class NLSI
"""

from regop import RegularizationOperator

from ..constraints import Constraint

class NLSI:
    """
    Implements a nonlinear least-squares problem of the form
    min_s ||func(s)||^2 + ||regop(s) - s_tilde||^2
    s. t. Cs >= d.
    """
    def __init__(self, func, jac, regop: RegularizationOperator, s_tilde, con: Constraint):
        self.func = func
        self.jac = jac
        self.regop = regop
        self.s_tilde = s_tilde
        self.con = con