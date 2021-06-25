"""
TODO: NEXT STEP: PROJECT THE NONLINEAR PROBLEM ON THE CONSTRAINTS
"""

import numpy as np

from operators import MultipliedOperator

from cgn.constraints import Constraint
from cgn.cnls import CNLS, CNLSSolution
from cgn.solvers.nlsi_solve import nlsi_solve
from cgn.solvers.nlsi import NLSI
from cgn.solvers.logger import Logger
from cgn.solvers.cgn_state import State
from cgn.utils import solve_upper_triangular


class ConstrainedGaussNewton:
    """
    This class manages the bounded Gauss-Newton iteration.
    """
    def __init__(self, cnls: CNLS):
        self._cnls = cnls
        # translate problem to nlsi problem
        self._nlsi, self._x1, self._S = self._remove_equality_constraints()
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
        # transform the initial guess
        s_start = self._reduce_starting_value(x_start)
        # solve the NLSI problem
        s_min, info = nlsi_solve(nlsi=self._nlsi, start=s_start, options=options)
        # translate the solution of the NLSI problem to the solution of the NLSEI problem.
        nlsei_solution = self._nlsi_to_nlsei_solution(s_min, info)
        # translate solution to solution of the equality constrained problem:
        return nlsei_solution

    def _reduce_starting_value(self, x_start):
        s_start = self._S.T @ x_start
        return s_start

    def _remove_equality_constraints(self):
        """
        Removes the equality constraint by transforming self._cnls
        """
        # if no equality constraint, then do nothing:
        if self._cnls.eqcon.cdim == 0:
            S = np.identity(self._cnls.dim)
            x1 = np.zeros(self._cnls.dim)
            # build nlsi from cnls
            s_tilde = self._cnls.regop.fwd(self._cnls.xbar)
            nlsi = NLSI(func=self._cnls.func, jac=self._cnls.jac, regop=self._cnls.regop, s_tilde=s_tilde,
                        con=self._cnls.incon)
            return nlsi, x1, S
        else:
            A = self._cnls.eqcon.mat
            b = self._cnls.eqcon.vec
            c1 = A.shape[0]
            QS, R = np.linalg.qr(A.T, mode="complete")
            R_nonzero = R[:c1, :]
            Q = QS[:, :c1]
            S = QS[:, c1:]
            if S.shape[1]==0:
                self._logger.log("WARNING: Solution completely determined by equality constraint.")
                # TODO: Write breakout function that in this case automatically wraps up the iteration.
            else:
                q = solve_upper_triangular(R_nonzero, b)
                x1 = Q @ q
                def func_tilde(s):
                    x = x1 + S @ s
                    return self._cnls.func(x)
                def jac_tilde(s):
                    jac = self._cnls.jac(x1 + S @ s)
                    jac_q = jac @ S
                    return jac_q
                regop_tilde = MultipliedOperator(regop=self._cnls.regop, q=S)
                s_tilde = self._cnls.regop.fwd(self._cnls.xbar - x1)
                if self._cnls.incon.cdim != 0:
                    C = self._cnls.incon.mat
                    d = self._cnls.incon.vec
                    C_tilde = C @ S
                    d_tilde = d - C @ x1
                else:
                    C_tilde = None
                    d_tilde = None
                # create a new CNLS that only has inequality constraints
                incon_new = Constraint(dim=S.shape[1], a=C_tilde, b=d_tilde)
                nlsi = NLSI(func=func_tilde, jac=jac_tilde, regop=regop_tilde, s_tilde=s_tilde, con=incon_new)
                return nlsi, x1, S

    def _nlsi_to_nlsei_solution(self, s_min, info):
        """
        Translates the solution of the reduced NLSI to the solution of the overall NLSEI
        """
        # translate minimizer
        x_min = self._x1 + self._S @ s_min
        nlsei_solution = self._build_solution(x_min=x_min, info=info)
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
