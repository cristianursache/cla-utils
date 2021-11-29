import numpy as np
from numpy import random
from cla_utils.exercises3 import householder_solve


def randomQ(m):
    """
    Produce a random orthogonal mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return Q: the mxm numpy array containing the orthogonal matrix.
    """
    Q, R = np.linalg.qr(random.randn(m, m))
    return Q


def randomR(m):
    """
    Produce a random upper triangular mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return R: the mxm numpy array containing the upper triangular matrix.
    """
    
    A = random.randn(m, m)
    return np.triu(A)


def backward_stability_householder(m):
    """
    Verify backward stability for QR factorisation using Householder for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        Q1 = randomQ(m)
        R1 = randomR(m)

        Q2, R2 = np.linalg.qr(Q1 @ R1)

        print('||Q2 - Q1|| = {}'.format(np.linalg.norm(Q2 - Q1)))
        print('||R2 - R1|| = {}'.format(np.linalg.norm(R2 - R1)))
        print('||A - Q2R2|| = {}'.format(np.linalg.norm(Q1 @ R1 - Q2 @ R2)))


def solve_R(R, b):
    """
    Solve the system Rx=b where R is an mxm upper triangular matrix 
    and b is an m dimensional vector.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :param x: an m-dimensional numpy array
    """

    m = len(R)
    x = np.zeros(m)

    x[-1] = b[-1] / R[-1, -1]
    for i in range(m-2, -1, -1):
        x[i] = (b[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    
    return x


def back_stab_solve_R(m):
    """
    Verify backward stability for back substitution for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        A = random.randn(m, m)
        R = np.triu(A)

        x = random.randn(m)
        y = R @ x

        x_tilde = solve_R(R, y)

        print(np.linalg.norm(R @ x_tilde - y) / np.linalg.norm(x_tilde - x))


def back_stab_householder_solve(m):
    """
    Verify backward stability for the householder algorithm
    for solving Ax=b for an m dimensional square system.

    :param m: the matrix dimension parameter.
    """
    for k in range(20):
        A = random.randn(m, m)
        x = random.randn(m)

        b = A @ x
        b = b.reshape((m, 1))

        x_tilde = householder_solve(A, b)

        print(np.linalg.norm(A @ x_tilde - b) / np.linalg.norm(x_tilde - x))
