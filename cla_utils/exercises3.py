import numpy as np
from scipy.linalg import solve_triangular


def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """

    m, n = A.shape
    if kmax is None:
        kmax = n

    R = A.copy()

    for k in range(kmax):
        x = R[k:, k]
        if x[0] != 0:
            alpha = (x[0] / np.abs(x[0])) * np.linalg.norm(x)
        else:
            alpha = np.linalg.norm(x)
        v = alpha * np.array([1 if i == 0 else 0 for i in range(len(x))]) + x
        v /= np.linalg.norm(v)
        R[k:, k:] -= 2 * np.dot(np.outer(v, np.conjugate(v)), R[k:, k:])

    return R


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """

    A_hat = A.copy()
    for i in range(np.shape(b)[1]):
        A_hat = np.column_stack((A_hat, b[:, i]))
    R_hat = householder(A_hat, kmax=np.shape(A)[1])

    x = np.zeros(np.shape(b))
    for i in range(np.shape(x)[1]):
        x[:, i] = solve_triangular(R_hat[:, :np.shape(A)[1]], R_hat[:, np.shape(A)[1]+i])

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the full QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """

    
    A_hat = np.concatenate((A, np.eye(np.shape(A)[0])), axis=1)
    QR = householder(A_hat, kmax=np.shape(A)[1])
    R = QR[:, :np.shape(A)[1]]
    Q = QR[:, np.shape(A)[1]:].T
    
    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    A_hat = np.column_stack((A, b))
    R_hat = householder(A_hat, kmax=np.shape(A)[1])
    x = solve_triangular(R_hat[:np.shape(A)[1], :np.shape(A)[1]], R_hat[:np.shape(A)[1], np.shape(A)[1]])

    return x
