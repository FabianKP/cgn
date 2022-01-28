"""
Contains class "LinesearchOptions"
"""


class LinesearchOptions:
    def __init__(self):
        self.maxiter = 52
        self.c = 0.5
        self.gamma_phi = 1e-5
        self.gamma_theta = 1e-5
        self.delta = 0.1
        self.s_theta = 1.1
        self.s_phi = 2.5
        self.eta = 1e-6
        self.gamma_h = 1e-10
        self.h0 = 1.
        self.ctol = 1e-15
        self.maxviol = 1000   # maximum error tolerance for constraint.
