from cla_utils import *
import numpy as np

def solve(A, b, lbd):
    """
    Given a matrix A, a vector b and a lambda, compute the stationary points (with
    respect to x) of the Lagrange multiplier phi corresponding to the least
    squares problem Ax = b with the constraint ||x|| = 1.

    :param A: an mxn-dimensional matrix
    :param b: an m-dimensional array
    :param lbd: scalar lambda

    :return x: solution to dphi / dx = 0
    """

    R = householder(A)
    z = - R @ A.T @ b 
    z /= lbd ** 2
    M = np.eye(np.shape(A)[0]) + (R @ R.T / lbd)
    x_star = householder_ls(M, z)
    x_star = R.T @ x_star
    x = (x_star + A.T @ b / lbd)

    return x

def exp_search_int(A, b, maxsteps=12):
    """
    Given a matrix A and a vector b, computes an interval of the form [1, lbd] 
    that contains a lambda for which ||x|| = 1.

    :param A: an mxn-dimensional matrix
    :param b: an m-dimensional array
    :param maxsteps: maximum number of iterations

    :return lbd: other end of the interval
    """

    lbd = 1

    norm = np.linalg.norm(solve(A, b, lbd))
    if norm > 1:
        for i in range(maxsteps):
            lbd *= 2
            if np.linalg.norm(solve(A, b, lbd)) > 1 and np.linalg.norm(solve(A, b, -lbd)) > 1:
                continue
            else:
                if np.linalg.norm(solve(A, b, lbd)) <= 1:
                    return lbd
                elif np.linalg.norm(solve(A, b, -lbd)) <= 1:
                    return -lbd
            

    elif norm < 1:
        for i in range(maxsteps):
            lbd *= 2
            if np.linalg.norm(solve(A, b, lbd)) < 1 and np.linalg.norm(solve(A, b, -lbd)) < 1:
                continue
            else:
                if np.linalg.norm(solve(A, b, lbd)) >= 1:
                    return lbd
                elif np.linalg.norm(solve(A, b, -lbd)) >= 1:
                    return -lbd

def binary_search(A, b, maxlbd, tolerance=1.0e-06):
    """
    Given a matrix A, a vector b and the endpoint of the interval [1, m], computes
    a lambda in the interval such that ||x|| is (arbitrarily) close to 1.

    :param A: an mxn-dimensional
    :param b: an m-dimensional array
    :param maxlbd: the endpoint of the interval [1, m] to do the binary search in
    :param tolerance: tolerance for abs(||x|| - 1)

    :return mid: lambda such that abs(||x|| - 1) < tolerance
    """

    if np.linalg.norm(solve(A, b, 1)) > 1:
        left = 1
        right = maxlbd
    else:
        left = maxlbd
        right = 1

    mid = (1 + maxlbd) / 2
    norm = np.linalg.norm(solve(A, b, mid))

    while abs(norm - 1) > tolerance:
        if norm > 1:
            left = mid
            mid = (left + right) / 2
        elif norm < 1:
            right = mid
            mid = (left + right) / 2
        norm = np.linalg.norm(solve(A, b, mid))

    return mid
