"""
Contains function "cls_solve".
"""
import numpy as np

from .cls import CLS
import qpsolvers as qp


def cls_solve(cls: CLS) -> np.ndarray:
    """
    Solves a constrained least-squares problem.
    :param cls: The problem object.
    :return: Of shape (n,). The minimizer of the constrained least-squares problem.
    """
    # Bring the CLS problem in the right format.
    r, s, g, h, a, b, lb = _bring_problem_in_right_form(cls)
    # Solve the problem with qpsolvers.
    x_min = qp.solve_ls(R=r, s=s, G=g, h=h, A=a, b=b, lb=lb)
    if x_min is None:
        raise RuntimeError("Linear solver could not find a solution.")
    return x_min


def _bring_problem_in_right_form(cls: CLS):
    """
    Brings the CLS problem in the right format so that qpsolvers.solve_ls can solve it.
    :param cls: CLS
    :return: r, s, g, h, a, b, lb
    """
    r = cls.h
    s = cls.y
    if cls.inequality_constrained:
        g = - cls.c
        h = - cls.d
    else:
        g = None
        h = None
    if cls.equality_constrained:
        a = cls.a
        b = cls.b
    else:
        a = None
        b = None
    if cls.bound_constrained:
        lb = cls.l
    else:
        lb = None
    return r, s, g, h, a, b, lb
