"""
Contains class "BoundedGaussNewton"
See the mathematical documentation for a formal description of the algorithm.
"""

import numpy as np
from time import time

from ..cls_solve import CLS, cls_solve
from ..cnls_solve.cnls_solution import CNLSSolution, OptimizationStatus
from ..regop import NullOperator
from .cnls import CNLS
from .linesearch_options import LinesearchOptions
from .solveroptions import Solveroptions
from .waechter_biegler import WaechterBiegler
from .logger import Logger
from .cgn_state import CGNState


class ConstrainedGaussNewton:
    """
    This class manages the bounded Gauss-Newton iteration.
    """
    def __init__(self, cnls: CNLS, options: Solveroptions, linesearch_options: LinesearchOptions):
        self._cnls = cnls
        # Read options
        self.maxiter = options.maxiter
        self.tol = options.tol
        self.ctol = options.ctol
        self._constraint_satisfied = True
        self.logfile = options.logfile
        self.timeout = options.timeout
        self._residual_list = []
        self._logger = Logger(verbosity=options.verbosity, filename=options.logfile)
        self._linesearch = WaechterBiegler(create_state=self._create_state, cnls=cnls, costfun=self._cost,
                                           options=linesearch_options)

    def solve(self, start) -> CNLSSolution:
        """
        Manages the outer loop of the iteration.
        :param start: starting value for the iteration
        :return: CNLSSolution
        """
        # Create initial state.
        state = self._create_state(start)
        current_cost = self._cost(state)
        self._add_residual(current_cost)
        status = OptimizationStatus.error
        t_start = time()
        k = -1
        self._logger.print_preamble(current_cost)
        self._logger.print_column_names()
        for k in range(1, self.maxiter + 1):
            t0 = time()
            # obtain step direction p by solving the linearized subproblem
            p = self._solve_subproblem(state)
            # obtain next position from the Wächter-Biegler linesearch
            state, current_cost, aborted, h = self._linesearch.next_position(state, p, self._cost_gradient(state))
            self._add_residual(current_cost)
            t = time() - t0
            constraint_violation = self._cnls.constraint_violation(state.x)
            self._constraint_satisfied = (constraint_violation <= self.ctol)
            # Do some logging.
            self._logger.print_iteration_info(k=k, cost=current_cost, cviol=constraint_violation, p=p, steplength=h,
                                              time=t)
            if aborted:
                status = OptimizationStatus.converged
                break
            if self._check_convergence():
                status = OptimizationStatus.converged
                break
            if self._check_time(time() - t_start):
                status = OptimizationStatus.timeout
        if k == self.maxiter:
            status = OptimizationStatus.maxout
        # check that MAP satisfies constraint
        if not self._cnls.satisfies_constraints(state.x, self.ctol):
            status = OptimizationStatus.constraint_violated
        cnls_solution = self._build_solution(state=state, niter=k, status=status)
        self._logger.print_epilogue(cnls_solution)
        return cnls_solution

    def _cost(self, state: CGNState):
        return 0.5 * (np.sum(np.square(state.h)) + np.sum(np.square(state.w))) / self._cnls.scale

    def _cost_gradient(self, state: CGNState):
        misfit_gradient = state.jac.T @ state.h
        reg_gradient = self._cnls.r.adj(state.w)
        return (misfit_gradient + reg_gradient) / self._cnls.scale

    def _posterior_precision(self, state: CGNState):
        """
        given the MAP estimator, computes an approximation of the square-root of the posterior covariance.
        :return: ndarray of shape (n,n)
        """
        j_min = state.jac
        p = self._cnls.r
        i_d = np.eye(p.dim)
        precision = p.adj(p.fwd(i_d)) + j_min.T @ j_min
        return precision

    def _build_solution(self, state, niter, status) -> CNLSSolution:
        """
        Builds an object of type BNLSSolution given the last state
        :param state: CGNState
        :param status: OptimizationStatus
            True if converged, otherwise False.
        :return: cgn.OptimizationSolution
        """
        precision = self._posterior_precision(state)
        # create costfunction that is returned as part of the solution
        current_cost = self._cost(state)
        # put maxiter in info
        solution = CNLSSolution(minimizer=state.x, precision=precision, min_cost=current_cost, niter=niter,
                                        status=status)
        return solution

    def _check_convergence(self):
        # check if all necessary convergence criteria are satisfied
        cost_converged = self._check_cost_convergence()
        constraint_satisfied = self._constraint_satisfied
        if cost_converged and constraint_satisfied:
            converged = True
        else:
            converged = False
        return converged

    def _check_cost_convergence(self):
        if len(self._residual_list) < 2:
            # cannot check convergence cause not enough residuals
            return False
        r_prev = self._residual_list[-2]
        r_last = self._residual_list[-1]
        scale = max(abs(r_prev), abs(r_last), 1.)
        cost_good = (0 <= (r_prev - r_last) / scale <= self.tol)
        return cost_good

    def _add_residual(self, r):
        self._residual_list.append(r)

    def _solve_subproblem(self, state: CGNState):
        """
        Solves the linearized subproblem
        min_deltax 0.5*||func(x) + jac(x) @ delta_x||^2 + 0.5*||P(x + p - m)||^2
        s. t. g + Gp = 0, h + H p >= 0.
        :return array_like
            the direction for the next step
        """
        linear_subproblem = self._linearize(state)
        delta_x = cls_solve(linear_subproblem)
        return delta_x

    def _linearize(self, state: CGNState) -> CLS:
        """
        Linearize the CNLS problem around the current state to obtain a linear subproblem.
        This linearized problem is simply

        .. math::
            min_p 0.5 ||F(x) + F'(x)p||^2 + 0.5 ||P(x + p - m)|^2
            s. t. G'(x)p = - G(x), H'(x) p \\geq - H(x).

        :param state:
        :return: LSIB
        """
        f = state.h
        j = state.jac
        x = state.x
        if self._cnls.equality_constrained:
            a = self._cnls.g_jac(x)
            b = - self._cnls.g(x)
        else:
            a = None
            b = None
        if self._cnls.inequality_constrained:
            c = self._cnls.h_jac(x)
            d = - self._cnls.h(x)
        else:
            c = None
            d = None
        if self._cnls.bound_constrained:
            lb = self._cnls.lb - state.x
            ub = self._cnls.ub - state.x
        else:
            lb = None
            ub = None
        if self._cnls.r is NullOperator:
            h = j
            y = - f
        else:
            p = self._cnls.r.mat
            h = np.concatenate((j, p), axis=0)
            y = - np.hstack((f, state.w))
        linear_subproblem = CLS(h=h, y=y, a=a, b=b, c=c, d=d, l=lb, u=ub, scale=self._cnls.scale)
        return linear_subproblem

    def _check_time(self, t):
        """
        Checks if the passed time in minutes is larger than timeout.
        :return: True, if passed time is larger. False, else.
        """
        if self.timeout is None:
            return False
        elif t / 60 > self.timeout:
            return True
        else:
            return False

    def _compute_w(self, x):
        if self._cnls.r is None:
            w = 0.
        else:
            w = self._cnls.r.fwd(x - self._cnls.m)
        return w

    def _create_state(self, x) -> CGNState:
        state = CGNState(self._cnls.jac, self._cnls.q)
        state.x = x
        state.w = self._compute_w(x)
        state.h = self._cnls.q.fwd(self._cnls.func(state.x))
        return state
