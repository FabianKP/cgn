"""
Contains classes "WaechterBiegler" and "WBState"
"""

import numpy as np

from ..cnls import CNLS
from .logger import Logger
from .linesearch import Linesearch
from ..utils import negative_norm
from .cgn_state import State


meps = np.finfo(float).eps


class WBState:
    """
    This class manages the state of the Waechter-Biegler line search, in the sense that it contains all the
    quantities that have to be computed for every new step length.
    """
    def __init__(self, state: State, phi, theta):
        self.state = state
        self.phi = phi
        self.theta = theta


class WaechterBiegler(Linesearch):
    """
    Implements the Lagrange-multiplier-free linesearch algorithm proposed by
    Wächter and Biegler in
    "Line Search Filter Methods for Nonlinear Programming: Motivation and Global Convergence" (2005).
    """
    def __init__(self, create_state, cnls: CNLS, options):
        """
        :param create_state: function
            A function that creates a valid state that the cost function 'phi' takes as argument.
        :param phi: function
            The cost function for the optimization problem.
        :param lb: ndarray
            The lower bound of the box constraint.
        :param ub: ndarray
            The upper bound of the box constraint.
        :param jac: ndarray
            Computes the Jacobian for the st
        :param options: A dictionary containing further options. The following options are used:
            verbose: If True, prints a warning if the minimal steplength is used,
            which can be evidence for infeasibility. If you see that warning, try to increase the
            parameter 'lsiter'.
            lsiter: Number of linesearch iterations.
            c, gamma_phi, delta, s_theta, s_phi, eta, gamma_h, h0: See mathematical documentation.
        """
        self._logger = Logger()
        self._set_options(options)
        self._create_state = create_state
        self._func = cnls.func
        self._jac = cnls.jac
        self._inequality_active = (cnls.incon.cdim > 0)
        self._con = cnls.incon
        self._lb = cnls.lb
        self._history = []

    def _phi(self, state: State):
        c = 0.5 *  np.linalg.norm(self._func(state.x))**2 + 0.5 * np.linalg.norm(state.w)**2
        return c

    def _theta(self, state: State):
        """
        The penalty function for the constraint.
        :param state: State
        :return:
        """
        penalty = 0
        if self._inequality_active:
            penalty += negative_norm(self._con.mat @ state.x - self._con.vec)
        if self._lb is not None:
            penalty += negative_norm(state.x - self._lb)
        return penalty

    def _set_options(self, linesearch_options):
        """
        Reads and sets the options for the line search.
        :param linesearch_options: dict
        """
        linesearch_options = self._check_options(linesearch_options)
        self._logger.verbose = linesearch_options.setdefault("verbose", False)
        self._lsiter = linesearch_options.setdefault("lsiter", 50)
        self._c = linesearch_options.setdefault("c", 0.5)
        self._gamma_phi = linesearch_options.setdefault("gamma_phi", 1e-5)
        self._gamma_theta = linesearch_options.setdefault("gamma_theta", 1e-5)
        self._delta = linesearch_options.setdefault("delta", 0.1)
        self._s_theta = linesearch_options.setdefault("s_theta", 1.1)
        self._s_phi = linesearch_options.setdefault("s_phi", 2.5)
        self._eta = linesearch_options.setdefault("eta", 1e-6)
        self._gamma_h = linesearch_options.setdefault("gamma_h", 1e-10)
        self._h0 = linesearch_options.setdefault("h0", 1.0)

    def _add_corner(self, state: WBState):
        """
        adds corner entry to filter
        """
        phi_w = state.phi
        theta_w = state.theta
        self._history.append([(1 - self._gamma_theta) * theta_w, phi_w - self._gamma_phi * theta_w])

    def _filter_allows(self, state: WBState):
        """
        :return: True if w satisfies filter condition.
        """
        phiw = state.phi
        thetaw = state.theta
        if all(ref[0] > thetaw or ref[1] > phiw for ref in self._history):
            return True
        else:
            return False

    def _compute_hmin(self, theta_now, m):
        """
        Computes the minimal step length for the line search.
        """
        if m < -meps:
            hmin = self._gamma_h * min((self._gamma_theta, - self._gamma_phi * theta_now / float(m),
                                        - self._delta * theta_now ** self._s_theta / float(m)))
        else:
            hmin = self._gamma_h * self._gamma_theta
        return hmin

    def _switching_condition(self, h, m, theta_now):
        """
        Checks if step size h triggers the switching condition.
        """
        if h*m < -meps and\
                (-h*m) * self._s_phi * h**(1 - self._s_phi) > self._delta * theta_now ** self._s_theta:
            return True
        else:
            return False

    def _armijo(self, state: WBState, h, m, phi_now):
        """
        Checks if proposed step satisfies Armijo condition.
        """
        if state.phi <= phi_now + self._eta * h * m:
            return True
        else:
            return False

    def _sufficient_decrease(self, state: WBState, phiNow, thetaNow):
        """
        Checks if proposed step satisfies sufficient decrease in
        the objective or in the constraint function.
        """
        if state.theta < (1 - self._gamma_theta)*thetaNow or \
                state.phi <= phiNow - self._gamma_phi*thetaNow:
            return True
        else:
            return False

    def next_position(self, state, delta_x, cost_gradient):
        """
        Implements the actual line search iteration.
        :param state: current state
        :param delta_x: proposed direction
        :return: next position w_next, computed with optimal steplength.
        """
        phi_now = self._phi(state)
        theta_now = self._theta(state)
        m = cost_gradient @ delta_x
        h = self._h0
        hmin = self._compute_hmin(theta_now, m)
        niter = 0
        new_state = self._create_wbstate(state.x + h * delta_x)
        found_h = False
        while (h > hmin+meps and niter<self._lsiter):
            niter += 1
            if self._switching_condition(h, m, theta_now):
                #self._logger.log("Switching condition triggered.")
                if self._armijo(new_state, h, m, phi_now):
                    #self._logger.log("Armijo satisfied.")
                    if self._filter_allows(new_state):
                        found_h = True
                        break
            elif self._sufficient_decrease(new_state, phi_now, theta_now):
                #self._logger.log("Sufficient decrease.")
                if self._filter_allows(new_state):
                    self._add_corner(new_state)
                    found_h = True
                    break
            h = self._c * h
            new_state = self._create_wbstate(state.x + h * delta_x)
        if not found_h:
            # if no good h was found, return old state and notify the calling iteration about failure
            aborted = True
            self._logger.log("Warning: Linesearch failed.")
            # new state is old state
            next_state = self._create_wbstate(state.x)
        else:
            aborted = False
            self._logger.log(f"Linesearch converged after {niter} iterations with h={h}")
            next_state = new_state
        return next_state.state, next_state.phi, aborted

    def _create_wbstate(self, x) -> WBState:
        """
        :param x: ndarray
        :return: WBState
            WBState object corresponding to x
        """
        state = self._create_state(x)
        wbstate = WBState(state, phi=self._phi(state), theta=self._theta(state))
        return wbstate
