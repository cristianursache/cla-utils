import numpy as np
from cla_utils.exercises3 import householder

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """

    A1 = A.copy()

    x = A1[:, 0]
    if x[0] != 0:
            alpha = np.sign(x[0]) * np.linalg.norm(x)
    else:
        alpha = np.linalg.norm(x)
    v = alpha * np.array([1 if i == 0 else 0 for i in range(len(x))]) + x
    v /= np.linalg.norm(v)

    A1[:, :] -= 2 * np.dot(np.outer(v, v),  A1[:, :])
    A1[:, :] -= 2 * np.outer(A1[:, :] @ v, v)

    return A1


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """

    m = len(A)

    for k in range(m-2):
        x = A[k+1:, k]
        if x[0] != 0:
            alpha = np.sign(x[0]) * np.linalg.norm(x)
        else:
            alpha = np.linalg.norm(x)
        v = alpha * np.array([1 if i == 0 else 0 for i in range(len(x))]) + x
        v /= np.linalg.norm(v)

        A[k+1:, k:] -= 2 * np.dot(np.outer(v, v), A[k+1:, k:])
        A[:, k+1:] -= 2 * np.outer(A[:, k+1:] @ v, v)


def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array
    
    :return Q: an mxm numpy array
    """

    m = len(A)

    Q = np.eye(m)

    for k in range(m-2):
        x = A[k+1:, k]
        if x[0] != 0:
            alpha = np.sign(x[0]) * np.linalg.norm(x)
        else:
            alpha = np.linalg.norm(x)
        v = alpha * np.array([1 if i == 0 else 0 for i in range(len(x))]) + x
        v /= np.linalg.norm(v)

        A[k+1:, k:] -= 2 * np.dot(np.outer(v, v), A[k+1:, k:])
        A[:, k+1:] -= 2 * np.outer(A[:, k+1:] @ v, v)
        Q[:, k+1:] -= 2 * np.outer(Q[:, k+1:] @ v, v)
    
    return Q


def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvalues and eigenvectors.

    :param H: an mxm numpy array

    :return ee: an m dimensional numpy array containing the eigenvalues of H
    :return V: an mxm numpy array whose columns are the eigenvectors of H
    """
    m, n = H.shape
    assert(m==n)
    assert(np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return the eigenvectors of A. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    Q = hessenbergQ(A)
    V = hessenberg_ev(A)
    return Q @ V
