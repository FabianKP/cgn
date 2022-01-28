"""
Contains classes "WaechterBiegler" and "WBState"
"""

import numpy as np

from .cnls import CNLS
from .linesearch_options import LinesearchOptions
from .cgn_state import CGNState

EPS = np.finfo(float).eps


class WBState:
    """
    This class manages the state of the Waechter-Biegler line search, in the sense that it contains all the
    quantities that have to be computed for every new step length.
    """
    def __init__(self, state: CGNState, phi, theta):
        self.state = state
        self.phi = phi
        self.theta = theta


class WaechterBiegler:
    """
    Implements the Lagrange-multiplier-free linesearch algorithm proposed by
    Wächter and Biegler in
    "Line Search LinearFilter Methods for Nonlinear Programming: Motivation and Global Convergence" (2005).
    """
    def __init__(self, create_state, cnls: CNLS, costfun: callable, options: LinesearchOptions):
        self.p = options
        self._phi = costfun
        self._create_state = create_state
        self._equality_constrained = cnls.equality_constrained
        self._inequality_constrained = cnls.inequality_constrained
        self._bound_constrained = cnls.bound_constrained
        self._g = cnls.g
        self._h = cnls.h
        self._lb = cnls.lb
        self._ub = cnls.ub
        self._history = []

    def _theta(self, state: CGNState):
        """
        The penalty function for the constraint.
        :param state: State
        :return:
        """
        penalty = 0
        x = state.x
        if self._equality_constrained:
            penalty += np.linalg.norm(self._g(x), ord=1)
        if self._inequality_constrained:
            penalty += self._negative_norm(self._h(x))
        if self._bound_constrained:
            penalty += self._negative_norm(x - self._lb)
            penalty += self._negative_norm(self._ub - x)
        # ignore penalty if less than tolerance:
        if penalty < self.p.ctol:
            penalty = 0
        return penalty

    def _add_corner(self, state: WBState):
        """
        adds corner entry to filter
        """
        phi_w = state.phi
        theta_w = state.theta
        self._history.append([(1 - self.p.gamma_theta) * theta_w, phi_w - self.p.gamma_phi * theta_w])

    def _filter_allows(self, state: WBState):
        """
        :return: True if w satisfies filter condition.
        """
        phiw = state.phi
        thetaw = state.theta
        constraint_ok = thetaw <= self.p.maxviol
        if all(ref[0] > thetaw or ref[1] > phiw for ref in self._history) and constraint_ok:
            return True
        else:
            return False

    def _compute_hmin(self, theta_now, m):
        """
        Computes the minimal step length for the line search.
        """
        if m < - EPS:
            hmin = self.p.gamma_h * min((self.p.gamma_theta, - self.p.gamma_phi * theta_now / float(m),
                                        - self.p.delta * theta_now ** self.p.s_theta / float(m)))
        else:
            hmin = self.p.gamma_h * self.p.gamma_theta
        return hmin

    def _switching_condition(self, h, m, theta_now):
        """
        Checks if step size h triggers the switching condition.
        """
        if h*m < -EPS and\
                (-h*m) * self.p.s_phi * h**(1 - self.p.s_phi) > self.p.delta * theta_now ** self.p.s_theta:
            return True
        else:
            return False

    def _armijo(self, state: WBState, h, m, phi_now):
        """
        Checks if proposed step satisfies Armijo condition.
        """
        if state.phi <= phi_now + self.p.eta * h * m:
            return True
        else:
            return False

    def _sufficient_decrease(self, state: WBState, phi_now, theta_now):
        """
        Checks if proposed step satisfies sufficient decrease in
        the objective or in the constraint function.
        """
        if state.theta < (1 - self.p.gamma_theta) * theta_now or \
                state.phi < phi_now - self.p.gamma_phi * theta_now:
            return True
        else:
            return False

    def next_position(self, state, p, cost_gradient):
        """
        Implements the actual line search iteration.
        :param state: current state
        :param p: proposed direction
        :param cost_gradient: (n,) array
            Gradient of cost function at current position.
        :return: next position w_next, computed with optimal steplength.
        """
        phi_now = self._phi(state)
        theta_now = self._theta(state)
        m = cost_gradient @ p
        h = self.p.h0
        niter = 0
        new_state = self._create_wbstate(state.x + h * p)
        found_h = False
        while niter < self.p.maxiter:
            niter += 1
            if self._switching_condition(h, m, theta_now):
                if self._armijo(new_state, h, m, phi_now):
                    if self._filter_allows(new_state):
                        found_h = True
                        break
            elif self._sufficient_decrease(new_state, phi_now, theta_now):
                if self._filter_allows(new_state):
                    self._add_corner(new_state)
                    found_h = True
                    break
            h = self.p.c * h
            new_state = self._create_wbstate(state.x + h * p)
        if not found_h:
            # if no good h was found, return old state and notify the calling iteration about failure
            aborted = True
            # new state is old state
            next_state = self._create_wbstate(state.x)
        else:
            aborted = False
            next_state = new_state
        return next_state.state, next_state.phi, aborted, h

    def _create_wbstate(self, x) -> WBState:
        """
        :param x: ndarray
        :return: WBState
            WBState object corresponding to x
        """
        state = self._create_state(x)
        wbstate = WBState(state, phi=self._phi(state), theta=self._theta(state))
        return wbstate

    @staticmethod
    def _negative_norm(x):
        x_neg = x.clip(max=0.)
        norm_neg = np.linalg.norm(x_neg, ord=1)
        return norm_neg
