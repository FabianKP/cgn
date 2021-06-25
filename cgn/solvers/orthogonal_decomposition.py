"""
Contains the functions qr_decomposition, translate_pivot, qr_with_pivoting
and orthogonal_decomposition.
"""


from numpy import eye, allclose
from numpy.linalg import qr
from scipy.linalg import qr as qr_with_pivot

from ..utils import is_square, make_matrix


def qr_decomposition(matrix):
    """
    wrapper for numpy's QR decomposition
    """
    return qr(matrix)


def translate_pivot(P):
    """
    Translates a permutation as it is returned by the pivoting qr-decomposition
    into a proper pivoting matrix
    :param P: permutation, i.e. vector of ints
    :return: pivoting matrix V, such that right-multiplying with V.T permutes columns
    """
    n = len(P)
    id = eye(n)
    V = id[P,:]
    return V


def qr_with_pivoting(matrix):
    """
    QR with pivoting.
    :param matrix: A (m, n)-matrix
    :return: Q, R, V such that matrix = Q R V.T.
        Q has shape (m,m), R has shape (m,n) and V has shape (n,n).
    """
    Q, R, P = qr_with_pivot(matrix, mode="full", pivoting=True)
    # translate pivoting matrix into real matrix
    V = translate_pivot(P)
    return Q, R, V


def orthogonal_decomposition(matrix):
    """
    Computes the orthogonal decomposition via rank-revealing QR
    to obtain matrix = [Q1, Q2] [R 0, 0 0] [U1, U2]^\top
    returns Q1, Q2, R, U1, U2, where Q2 and U2 might be None.
    See mathematical documentation.
    :param matrix: A (m, n) matrix. Might be rank deficient.
    :return: Q1, Q2, R, U1, U2, where Q2 and U2 might be None, such that
            matrix = [Q1, Q2] [R, 0; 0, 0] [U1.T; U2.T]. The matrices have shape
            (m,r), (m,m-r), (r,r), (r,n), (m-r,n), where r is the rank of 'matrix'.
            If m-r=0, then Q2 and P2 are None.
    """
    # first, obtain pivoting QR decomposition
    U, M, V = qr_with_pivoting(matrix.T)
    # delete zero rows
    nonzero_row_indices = [i for i in range(M.shape[0]) if not allclose(M[i, :], 0.0)]
    M_nonzero = M[nonzero_row_indices, :]
    if is_square(M_nonzero):
        # This means R is already a nonsingular, square upper triangular matrix
        # and we are done.
        R = M_nonzero
        Q = V
    else:
        W, R = qr_decomposition(M_nonzero.T)
        # again, delete zero rows of R
        nonzero_row_indices = [i for i in range(R.shape[0]) if not allclose(R[i, :], 0.0)]
        R = R[nonzero_row_indices, :]
        assert is_square(R), "R must be square"
        Q = V @ W
    rank = R.shape[0]
    Q1 = Q[:,:rank]
    Q2 = Q[:,rank:]
    S1 = U[:,:rank]
    S2 = U[:,rank:]
    # turn empty matrices into None:
    Q1, Q2, R, S1, S2 = make_matrix((Q1, Q2, R, S1, S2))
    return Q1, Q2, R, S1, S2