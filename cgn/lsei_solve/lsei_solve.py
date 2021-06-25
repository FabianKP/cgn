"""
Algorithms to solve linear least squares problem with equality and inequality constraints.
Contains functions lseq, lsi, lsei, reduce_eqcon, ldp, nnls and lsq
"""

import numpy as np
from scipy.optimize import nnls as scipy_nnls
from scipy.optimize import lsq_linear as scipy_lsq

from .ls_problem import LSProblem, Type

from ..solvers.orthogonal_decomposition import orthogonal_decomposition
from ..utils import invert_triangular, solve_upper_triangular


def solve_least_squares(lsp: LSProblem):
    if lsp.type == Type("ls"):
        x_min = solve_ls(lsp)
    elif lsp.type == Type("lse"):
        x_min = solve_lse(lsp)
    elif lsp.type == Type("lsi"):
        x_min = solve_lsi(lsp)
    elif lsp.type == Type("lsb"):
        x_min = solve_lsb(lsp)
    else:
        x_min = solve_lsei(lsp)
    return x_min


def solve_ls(ls: LSProblem):
    """
    Solves the unconstrained least-squares problem
    min 0.5*||Ax-b||^2
    :return: The minimizer x.
    """
    assert ls.type == Type("ls")
    x = scipy_lsq(ls.A, ls.b).x
    return x


def solve_lse(lse: LSProblem):
    """
    Solves the _constrained least-squares problem
    min_x 0.5*||Ax-b||^2 s.t. C@x = d
    :return: the minimizer x_min
    """
    assert lse.type == Type("lse")
    x1, P2, ls = reduce_eqcon(lse)
    p2_min = solve_ls(ls)
    x_min = P2 @ p2_min + x1
    return x_min


def reduce_eqcon(lse: LSProblem, verbose=False):
    """
    Reduces the fully-_constrained problem
    min_z 0.5*||Az-b||^2 s.t. Cz=d, Ez >= f
    to the inequality-_constrained problem
    min_p 0.5*||A_t p - b_t||^2 s.t. E_t p >= f_t.
    :param verbose: A Boolean variable. If True, it prints out a warning
        if the equality constraint is infeasible.
    :return: LSProblem
    if problem.type == "lse", return type is "ls".
    if problem.type == "lsei", return type is "lsi"
    """
    assert lse.type == Type("lse") or lse.type == Type("lsei")
    Q1, Q2, R, P1, P2 = orthogonal_decomposition(lse.C)
    p1 = solve_upper_triangular(R, Q1.T @ lse.d)
    x1 = P1 @ p1
    if verbose and not np.allclose(Q2.T @ lse.d, 0):
        print("WARNING: Infeasibility of equality constraints detected. Ignoring infeasible constraints.")
    A_t = lse.A @ P2
    b_t = lse.b - lse.A @ x1
    if lse.type == Type("lsei"):
        E_t = lse.E @ P2
        f_t = lse.f - lse.E @ x1
    else:
        E_t = None
        f_t = None
    # make returned problem
    ls_no_equality_constraints = LSProblem(A=A_t, b=b_t, E=E_t, f=f_t)
    return x1, P2, ls_no_equality_constraints


def eqcon_to_p(C):
    """
    Auxiliary method for equality-constrained covariance
    :param C: matrix of shape (m,n) with m <= n
    :return: P1, P2
    """
    Q1, Q2, R, P1, P2 = orthogonal_decomposition(C)
    return P1, P2


def solve_lsei(lsei: LSProblem, verbose=False):
    """
    Solves the fully _constrained least-squares problem
    min_x 0.5 * ||Ax-b||^2 s.t. Cx=d, Ex>=f.
    to the inequality-_constrained problem
    :param verbose: A Boolean. If True, it prints out a warning if a constraint is infeasible.
    :return: The solution vector z_min.
    """
    assert lsei.type == Type("lsei")
    x1, P2, lsi = reduce_eqcon(lsei)
    p2_min = solve_lsi(lsi)
    z_min = x1 + P2 @ p2_min
    return z_min


def solve_lsb(lsb: LSProblem):
    """
    Solves the bound _constrained least-squares problem
    min 0.5*||Ax-b||^2 s.t. lb <= x
    :param A: (m,n) array
    :param b: (m,) array
    :param lb: (n,) array denoting the lower bound. Set lb[i]=-np.inf to turn of bound on i-th coefficient.
    :param ub: (n,) array denoting the upper bound. Set ub[i]=np.inf to turn of bound on the i-th coefficient.
    :return: the minimizer x of shape (n,)
    """
    assert lsb.type == Type("lsb")
    ub = np.inf * np.ones_like(lsb.lb)
    x = scipy_lsq(lsb.A, lsb.b, bounds=(lsb.lb, ub)).x
    return x


def solve_lsi(lsi: LSProblem):
    """
    Solves the LSI problem: min 0.5*||Ax-b||^2 s.t. Ex>=f
    :return: ndarray
    """
    assert lsi.type == Type("lsi")
    U1, U2, T, V1, V2 = orthogonal_decomposition(lsi.A)
    if V2 is not None:
        V = np.concatenate((V1, V2), axis=1)
    else: V = V1
    T_inv = invert_triangular(T)
    K = lsi.E @ V @ T_inv
    l = lsi.f - K @ U1.T @ lsi.b
    u = ldp(K, l)
    x = V @ T_inv @ (u + U1.T @ lsi.b)
    return x


def ldp(K, l):
    """
    Solves min 0.5*||u||^2 s.t. Ku >= l
    :return: The minimizer x_min of shape (n,)
    """
    n = K.shape[1]
    M = np.vstack((K.T, l))
    e = np.zeros(n+1)
    e[n] = 1.0
    r = nnls(M, e)
    u = - r[:n] / r[n]
    return u


def nnls(A, b):
    """
    Solves the NNLS problem: min 0.5*||Ax-b||^2 s.t. x>=0.
    :return: The vector res = Ax - b, where x is the minimizer of the NNLS problem.
    """
    nnls_maxiter = 10 * max(A.shape[0], A.shape[1])
    x, r = scipy_nnls(A, b, maxiter=nnls_maxiter)
    res = A @ x - b
    return res


