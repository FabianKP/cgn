"""
Contains function NLSI Solver.
"""

import numpy as np
from time import time

from .nlsi import NLSI
from .logger import Logger
from .waechter_biegler import WaechterBiegler
from ..lsei_solve import solve_lsi
from .cgn_state import State


def nlsi_solve(nlsi: NLSI, start, options):
    """
    Solves NLSI problem
    :param nlsi: NLSI
    :param start: ndarray
        Initial guess for NLSI problem.
    :param options:
    :return:
    """
    solver = NLSISolver(nlsi, start, options)
    solution = solver.solve()
    return solution


class NLSISolver:
    """
    This class manages the bounded Gauss-Newton iteration.
    """
    def __init__(self, nlsi: NLSI, start, options):
        self._nlsi = nlsi
        self._start = start
        self._handle_options(options)
        self._residual_list = []
        self._gradient_list = []
        self._logger = Logger()
        self._logger.verbose = self.verbose

    def solve(self):
        """
        Solves an inequality-constrained nonlinear least-squares problem.
        :return:
        """
        # create initial state
        state = self._create_state(self._start)
        current_cost = self._cost(state)
        self._add_residual(current_cost)
        self._logger.log(f"Cost at starting value: {current_cost}")
        # initialize Wächter-Biegler linesearch for reduced problem
        linesearch = WaechterBiegler(create_state=self._create_state, nlsi=self._nlsi, options=self._options)
        # do the Gauss-Newton iteration:
        converged = False
        t_start = time()
        k = 0
        aborted = False
        for k in range(self.maxiter):
            self._logger.newline()
            self._logger.log(f"ITERATION {k + 1}")
            t0 = time()
            # obtain delta_x by solving the linearized subproblem
            delta_s = self._solve_subproblem(state)
            print(f"delta_s = {delta_s}")
            # obtain next position from the Wächter-Biegler linesearch
            state, current_cost, aborted = linesearch.next_position(state, delta_s, self._cost_gradient(state))
            self._add_residual(current_cost)
            t1 = time()
            self._logger.log(f"Computation time: {t1 - t0:6f} seconds.")
            self._logger.log(f"||Delta||={np.linalg.norm(delta_s)}")
            self._logger.log(f"Current cost: {self._residual_list[-1]}")
            # check_convergence at the next position
            if aborted:
                break
            if self._check_convergence(delta_s):
                converged = True
                break
            if self._check_time(time() - t_start):
                converged = True
                break
        self._logger.newline()
        if converged:
            self._logger.log(f"The constrained Gauss-Newton iteration converged after {k + 1} iterations.")
        elif aborted:
            self._logger.log(f"Constrained Gauss-Newton stopped: Wächter-Biegler linesearch could not find"
                             f" an admissible step.")
        else:
            self._logger.log(
                "The constrained Gauss-Newton iteration stopped as it reached maximum number of iterations.")
        self._logger.newline()
        # compute all interesting quantities for the solution
        info = {"niter": k+1}
        return state.s, info

    def _check_convergence(self, delta_x):
        # check if all necessary convergence criteria are satisfied
        delta_convergence = self._check_delta_convergence(delta_x)
        emergency_stop = self._check_emergency_stop()
        if delta_convergence or emergency_stop:
            converged = True
        else:
            converged = False
        return converged

    def _check_delta_convergence(self, delta_x):
        if np.linalg.norm(delta_x) < self.tol:
            print("Delta_good")
            return True
        else:
            return False

    def _check_emergency_stop(self):
        if len(self._residual_list) < 2:
            # cannot check convergence cause not enough residuals
            return False
        emergency_tol = 1e-16
        r_prev = self._residual_list[-2]
        r_last = self._residual_list[-1]
        scale = max(abs(r_prev), abs(r_last), 1.)
        cost_good = (0 <= (r_prev - r_last) / scale <= emergency_tol)
        if cost_good:
            print("Emergency stop!")
        return cost_good

    def _add_residual(self, r):
        self._residual_list.append(r)

    def _add_gradient(self, r):
        self._gradient_list.append(r)

    def _handle_options(self, options):
        if options is None:
            options = {}
        else:
            assert isinstance(options, dict)
        self._options = options
        self.tol = options.setdefault("tol", 1e-3)
        self.ctol = options.setdefault("ctol", None)
        if self.ctol is None:
            self._check_constraint = False
        else:
            self._check_constraint = True
        self.gtol = options.setdefault("gtol", None)
        if self.gtol is None:
            self._check_gradient = False
        else:
            self._check_gradient = True
        self.maxiter = options.setdefault("maxiter", 1000)
        self.verbose = options.setdefault("verbose", False)
        self._timeout = options.setdefault("timeout", None)
        if self._timeout is not None:
            assert isinstance(self._timeout, int), "Timeout must be int."

    def _solve_subproblem(self, state: State):
        """
        Solves the linearized subproblem
        min_deltax 0.5*||func(x) + jac(x) @ delta_x||^2 + 0.5*||p(delta_x + x - x_bar)||^2
        s. t. c @ delta_x >= d - c @ x
        :return numpy vector
            the direction for the next step
        """
        # setup LSI (least-squares with inequality constraints)
        F = state.f
        J = state.j()
        P = self._nlsi.regop.mat
        G = np.concatenate((J, P), axis=0)
        h = - np.hstack((F, state.w))
        E = self._nlsi.con.mat
        f = self._nlsi.con.vec - E @ state.s
        delta_x = solve_lsi(G=G, h=h, E=E, f=f)
        return delta_x

    @staticmethod
    def _cost(state: State):
        return 0.5 * np.linalg.norm(state.f) ** 2 + 0.5 * np.linalg.norm(state.w) ** 2

    def _cost_gradient(self, state: State):
        misfit_gradient = state.j().T @ state.f
        reg_gradient = self._nlsi.regop.mat.T @ state.w
        return misfit_gradient + reg_gradient

    def _check_time(self, t):
        """
        Checks if the passed time in minutes is larger than timeout.
        :return: True, if passed time is larger. False, else.
        """
        if self._timeout is None:
            return False
        elif t / 60 > self._timeout:
            if self.verbose:
                print("The iteration terminated due to timeout.")
            return True
        else:
            return False

    def _compute_w(self, x):
        return self._nlsi.regop.fwd(x) - self._nlsi.s_tilde

    def _create_state(self, s):
        state = State(self._nlsi.jac)
        state.s = s
        state.w = self._compute_w(s)
        state.f = self._nlsi.func(state.s)
        return state