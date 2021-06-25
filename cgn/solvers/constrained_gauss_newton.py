"""
Contains class "BoundedGaussNewton"
See the mathematical documentation for a formal description of the algorithm.
"""

import numpy as np
from time import time

from ..lsei_solve.lsei_solve import solve_least_squares
from ..lsei_solve.ls_problem import LSProblem
from ..cnls import CNLS, CNLSSolution
from .waechter_biegler import WaechterBiegler
from .logger import Logger
from .cgn_state import State


class ConstrainedGaussNewton:
    """
    This class manages the bounded Gauss-Newton iteration.
    """
    def __init__(self, cnls: CNLS):
        self._cnls = cnls
        self._residual_list = []
        self._gradient_list = []
        self._logger = Logger()

    def iterate(self, x_start, options):
        """
        Manages the outer loop of the iteration.
        :param x_start: starting value for the iteration
        :param options: A dictionary of _solver options.
            'tol': float > 0
                Tolerance for cost function. If None is provided, it defaults to 1e-10
            'ctol': float > 0
                Tolerance for constraint violation. If None is provided, arbitrary constraint violation is tolerated.
            'gtol': float > 0
                Tolerance for reduced gradient. If None is provided, abitrarily high reduced gradients are tolerated.
            'maxiter': int
                Maximum number of iterations. Defaults to 1000.
            'verbose': bool
                If True, info is displayed to console.
            'timeout: Number of minutes until the method times out. There is no default timeout.
            Also, additional options for the linesearch can be written in 'options'. See help(Waechter-Biegler)
        :return: CNLSSolution
        """
        self._handle_options(options)
        self._logger.verbose = self.verbose
        # transform the initial guess
        state = self._create_state(x_start)
        current_cost = self._cost(state)
        self._add_residual(current_cost)
        self._logger.log(f"Cost at starting value: {current_cost}")
        # initialize Wächter-Biegler linesearch for reduced problem
        linesearch = WaechterBiegler(create_state=self._create_state, cnls=self._cnls, options=options)
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
            delta_x = self._solve_subproblem(state)
            # obtain next position from the Wächter-Biegler linesearch
            state, current_cost, aborted = linesearch.next_position(state, delta_x, self._cost_gradient(state))
            self._add_residual(current_cost)
            t1 = time()
            self._logger.log(f"Computation time: {t1 - t0:6f} seconds.")
            self._logger.log(f"||Delta_x||={np.linalg.norm(delta_x)}")
            self._logger.log(f"Current cost: {self._residual_list[-1]}")
            # check_convergence at the next position
            if aborted:
                break
            if self._check_convergence(delta_x):
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
        info = {"niter": k+1}
        nlsei_solution = self._build_solution(state.x, info=info)
        # translate solution to solution of the equality constrained problem:
        return nlsei_solution

    @staticmethod
    def _cost(state: State):
        return 0.5 * np.linalg.norm(state.f) ** 2 + 0.5 * np.linalg.norm(state.w) ** 2

    def _posterior_precision(self, x_min):
        """
        given the MAP estimator, computes an approximation of the square-root of the posterior covariance.
        :return: ndarray of shape (n,n)
        """
        j_min = self._cnls.jac(x_min)
        p = self._cnls.regop.mat
        precision = p.T @ p + j_min.T @ j_min
        return precision

    def _build_solution(self, x_min, info) -> CNLSSolution:
        """
        Builds an object of type BNLSSolution given the last state
        :param x_min: ndarray
        :param info: dict
        :return: BNLSSolution
        """
        precision = self._posterior_precision(x_min)
        # create costfunction that is returned as part of the solution
        def costfunction(x):
            w = self._cnls.regop.fwd(x-self._cnls.xbar)
            misfit = np.linalg.norm(self._cnls.func(x)) ** 2
            regterm = np.linalg.norm(w) ** 2
            return 0.5 * (misfit + regterm)
        current_cost = costfunction(x_min)
        # put maxiter in info
        solution = CNLSSolution(minimizer=x_min, precision=precision, min_cost=current_cost,
                                costfun=costfunction, info=info)
        return solution

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
        P = self._cnls.regop.mat
        if self._cnls.eqcon.cdim == 0:
            C = None
            d = None
        else:
            C = self._cnls.eqcon.mat
            d = self._cnls.eqcon.vec - C @ state.x
        if self._cnls.incon.cdim == 0:
            E = None
            f = None
        else:
            E = self._cnls.incon.mat
            f = self._cnls.incon.vec - E @ state.x
        if self._cnls.lb is None:
            lb = None
        else:
            lb = self._cnls.lb - state.x
        linear_subproblem = LSProblem(A = np.concatenate((J, P), axis=0),
                                      b = - np.hstack((F, state.w)),
                                      C = C,
                                      d = d,
                                      E = E,
                                      f = f,
                                      lb = lb)
        delta_x = solve_least_squares(linear_subproblem)
        return delta_x

    def _cost_gradient(self, state: State):
        misfit_gradient = state.j().T @ state.f
        reg_gradient = self._cnls.regop.mat.T @ state.w
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
        return self._cnls.regop.fwd(x-self._cnls.xbar)

    def _create_state(self, x):
        state = State(self._cnls.jac)
        state.x = x
        state.w = self._compute_w(x)
        state.f = self._cnls.func(state.x)
        return state
