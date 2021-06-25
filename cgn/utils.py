"""
Contains auxiliary functions is_vector, matrix_root, find_w, w_nonzero, negative_norm, make_2d, make_matrix,
is_square, is_matrix, solve_spd, solve_upper_triangular and invert_triangular.
"""

from numpy import finfo, where, identity
import numpy as np
from numpy.random import normal
from scipy.linalg import solve, solve_triangular, sqrtm, eigh

meps = finfo(float).eps


def is_vector(vec):
    """
    Checks if vec is a vector of shape (n,)
    :param vec:
    :return: True if yes, False if no.
    """
    isvec = True
    try:
        if vec.ndim != 1:
            isvec = False
    except:
        isvec = False
    return isvec


def matrix_root(A):
    """
    Computes the symmetric matrix root.
    :param A: A (n,n) array.
    :return: A matrix B of shape (n,n) such that B @ B.T= A.
    """
    return sqrtm(A)


def find_w(x, x0, S):
    """
    Solves S @ w = x - x0.
    :param x: A vector of shape (n,)
    :param x0: A vector of shape (n,)
    :param S: A matrix of shape (n,n)
    :return: A vector w of shape (n,)
    """
    try:
        w = solve_spd(S, x - x0)
        return w
    except:
        print("Error: x-_mean is not in span of _covroot")


def w_nonzero(w, eps=1e-10):
    """
    If |w[i]| < eps, then this coefficient
    is substituted with Gaussian noise of scale 10.*eps
    :param w: ndarray of shape (n,)
    :return: A new vector w with |w[i]|>eps for all i.
    """
    eps = 1e-2
    # makes sure that no component of x=_mean+_covroot@wstart is zero
    smallIndices = where(abs(w) <= eps)
    random = eps*normal(size=smallIndices[0].size)
    w[smallIndices] = random
    return w


def negative_norm(x):
    """
    Computes sum of the negative parts of a given vector
    :param x: numpy vector
    :return: sum of negative parts of the coefficients of x
    """
    negindices = where(x < 0)[0]
    return -sum(x[negindices])


def make_2d(A):
    """
    If A is a vector, turn it into a (n,1)-array.
    If A is a twodimensional array, do nothing.
    Else, raise Exception.
    :param A: numpy array
    :return: 2-dimensional numpy array
    """
    if A is None:
        return None
    elif A.ndim == 1:
        A.shape = (len(A), 1)
    elif A.ndim==2 and A.shape[1]==0:
        return None
    elif A.ndim > 2:
        raise Exception("Error: v has more than 2 dimensions?")
    return A


def make_matrix(maybetuple):
    """
    ensures
    :param maybetuple:
    :return:
    """
    if isinstance(maybetuple, tuple):
        matrixlist = []
        for A in maybetuple:
            matrixlist.append(make_2d(A))
        return tuple(matrixlist)
    else:
        matrix = make_2d(maybetuple)
        return matrix


def is_matrix(matrix):
    """
    Returns True if matrix.ndim == 2.
    """
    try:
        if matrix.ndim == 2:
            return True
        else:
            return False
    except:
        return False


def is_square(matrix):
    """
    Returns true if matrix.shape[0] = matrix.shape[1].
    """
    try:
        if matrix.shape[0] == matrix.shape[1]:
            return True
        else:
            return False
    except:
        return False


def solve_spd(matrix, rhs):
    """
    Wrapper for linear system _solver for symmetric positive definite matrices
    :param matrix: must be symmetric positive definite.
    :param rhs: right-hand side
    :return: (n,1)-array x, where x is such that matrix @ x = rhs
    """
    x = solve(matrix, rhs, assume_a='pos')
    return x


def solve_upper_triangular(matrix, rhs):
    """
    Wrapper for linear system _solver for upper triangular matrices
    :param matrix: must be upper triangular (and square)
    :param rhs: right-hand side
    :return: (n,)-array x, where x is such that matrix @ x = rhs
    """
    x = solve_triangular(matrix, rhs)
    return x


def invert_triangular(matrix):
    """
    Inversion of an upper triangular matrix based on scipy.linalg.solve_triangular
    """
    return solve_upper_triangular(matrix, identity(matrix.shape[0]))

def sqrt_r(p, alpha):
    """
    Computes
    :param p: symmetric positive semidefinite matrix
    :param alpha: strictly positive float
    :return: an asymmetric square-root of (a + alpha*id)^(-1), and the corresponding precision
    """
    d, u = eigh(p)
    sqrt = u * np.divide(1, np.sqrt(d + alpha))
    prec = np.sqrt(d + alpha) * u.T
    return sqrt, prec
