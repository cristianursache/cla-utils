from operator import invert
from cla_utils import *
import numpy as np


def householder_rv(A, kmax=None):
    """
    Given a matrix A, computes its associated Rv matrix (the upper triangular 
    block is the matrix R in the full QR decomposition and the columns below
    the diagonal are the vectors v_k used for forming Q_k).

    :input A: an mxn-dimensional matrix
    :input kmax: the number of columns of A to reduce to upper triangular

    :return Rv: the Rv matrix of A
    """

    m, n = A.shape
    if kmax is None:
        kmax = n

    Rv = A.copy()

    for k in range(kmax):
        x = Rv[k:, k]
        if x[0] != 0:
            alpha = np.sign(x[0]) * np.linalg.norm(x)
        else:
            alpha = np.linalg.norm(x)

        v = alpha * np.array([1 if i == 0 else 0 for i in range(len(x))]) + x
        v /= np.linalg.norm(v)
        Rv[k:, k:] -= 2 * np.dot(np.outer(v, v), Rv[k:, k:])
          

        if k == kmax - 1 and m == n:
          pass
        
        else:
          vsplit = np.hsplit(v, len(v))
          vstack = np.vstack(vsplit[1:]) 
          Rv[k+1:, k:k+1] = vstack
        
    return Rv

def invertQ_rv(Rv, b):
    """
    Given the Rv matrix of a matrix A and a vector b, computes Q^*b.

    :input Rv: an mxn-dimensional matrix, the Rv matrix of a matrix A
    :input b: an m-dimensional vector

    :return Qstarb: the vector Q^*b
    """
    
    m, n = Rv.shape

    Qstarb = np.vstack(np.hsplit(b, len(b)))

    for k in range(n):
        if k == n - 1 and m == n:
          diag = np.sign([Rv[k, k]]) * np.sqrt(1 - sum(i ** 2 for i in Rv[k+1:, k:k+1]))
          Qstarb[k:m] -= 2 * np.vstack(diag) * (diag @ Qstarb[k:m])

        else:
          diag = -np.sign([Rv[k, k]]) * np.sqrt(1 - sum(i ** 2 for i in Rv[k+1:, k:k+1]))
          v = Rv[k+1:, k:k+1]
          v = np.concatenate((diag, np.hstack(v)))
          Qstarb[k:m] -= 2 * np.vstack(v) * (v @ Qstarb[k:m])

    return Qstarb

def solve_rv(Rv, b):
    """
    Given the Rv matrix of a matrix A and a vector b, computes the least
    sqares solution to Ax = b.

    :input Rv: an mxn-dimensional matrix, the Rv matrix of a matrix A
    :input b: an m-dimensional vector

    :return sol: the solution to Ax = b
    """
    
    n = Rv.shape[1]

    Qstarb = invertQ_rv(Rv, b)

    R = np.zeros((n, n))
    for k in range(n):
      R[k:k+1, k:] = Rv[k:k+1, k:]

    sol = np.hstack(solve_triangular(R, Qstarb[:n]))

    return sol
