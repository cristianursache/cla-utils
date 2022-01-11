import numpy as np
from cla_utils import pure_QR

# add docstrings + examples for n = 2, 3

def createA(n):
    """
    Given n, creates a 2nx2n matrix of the type A.

    :input n: an integer

    :return A: a 2nx2n-dimensional numpy array
    """

    A = np.zeros((2*n, 2*n))
    A += np.diag(np.ones(2*n - 1), 1)
    A -= np.diag(np.ones(2*n - 1), -1)
    return A

A2 = createA(2)
Ak2 = pure_QR(A2, 100, .001)
print(Ak2)

A5 = createA(5)
Ak5 = pure_QR(A5, 1000, .001)
print(Ak5)

def ev_A(n, nit):
    """
    Given n and the number of iterations, computes the eigenvalues
    of a 2nx2n matrix of the type A.

    :input n: an integer
    :input nit: an integer (number of iterations)

    :return e: a 2n-dimensional numpy array containing the
    eigenvalues of A.
    """

    A = createA(n)
    e = np.zeros(2 * n, dtype=complex)
    # tol does nothing in this case (matrix is not upper-triangular)
    Ak = pure_QR(A, nit, .001)
    for i in range(n):
        e[i] = 1j *  Ak[2*i, 2*i + 1]
    e[n:] = -e[:n]
    return e

print(np.linalg.eig(A2)[0])
print(ev_A(2, 100))

print(np.linalg.eig(A5)[0])
print(ev_A(5, 1000))

def createB(n):
    """
    Given n, creates a 2nx2n matrix of the type B.

    :input n: an integer

    :return B: a 2nx2n-dimensional numpy array
    """

    B = np.zeros((2*n, 2*n))
    B += 2 * np.diag(np.ones(2*n - 1), 1)
    B -= np.diag(np.ones(2*n - 1), -1)
    return B

def ev_B(n, nit):
    """
    Given n and the number of iterations, computes the eigenvalues
    of a 2nx2n matrix of the type A.

    :input n: an integer
    :input nit: an integer (number of iterations)

    :return e: a 2n-dimensional numpy array containing the
    eigenvalues of A.
    """

    B = createB(n)
    e = np.zeros(2 * n, dtype=complex)
    # tol does nothing in this case (matrix is not upper-triangular)
    Bk = pure_QR(B, nit, .001)
    for i in range(n):
        det = -Bk[2*i, 2*i + 1] * Bk[2*i + 1, 2*i]
        e[i] = 1j * np.sqrt(det)
    e[n:] = -e[:n]
    return e

B2 = createB(2)
B5 = createB(5)

print(np.linalg.eig(B2)[0])
print(ev_B(2, 100))

print(np.linalg.eig(B5)[0])
print(ev_B(5, 1000))
