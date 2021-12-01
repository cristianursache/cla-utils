import numpy as np

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.

    :return pij: an m-dimensional integer array describing the permutation \
    i.e. (Px)[i] = x[p[i]]
    """

    pij = p
    pij[i], pij[j] = pij[j], pij[i]

    return pij


def LUP_inplace(A):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.

    :param A: an mxm-dimensional numpy array

    :return p: an m-dimensional integer array describing the permutation
    associated with the decomposition PA = LU.
    """
                     
    m = len(A)
    p = [i for i in range(m)]

    for k in range(m-1):
        i = np.argmax(abs(A[k:, k])) + k
        A[[i, k]] = A[[k, i]]
        p = perm(p, i, k)
        lk = np.zeros(m)
        lk[k+1:] = -A[k+1:, k] / A[k, k]
        ek = np.zeros(m)
        ek[k] = 1
        outer = np.outer(lk, ek)
        A[k:, k:] += outer[k:, k:] @ A[k:, k:]
        A -= outer

    return p


def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """
                     
    raise NotImplementedError

def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """
                     
    raise NotImplementedError
