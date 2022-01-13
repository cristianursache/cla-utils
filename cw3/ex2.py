import numpy as np
from cla_utils import hessenberg, pure_QR

def createAij(n):
    i, j = np.indices((n, n))
    A = 1 / (1 + i + j)
    return A

# Eigenvalues of A using np.linalg.eig
A = createAij(5)
print(np.linalg.eig(A)[0])

# Eigenvalues of A using the new termination condition
hessenberg(A)
Ak = pure_QR(A, 10000, tridiagonal=True)[0]
print(np.diag(Ak, 0))

def modified_QR(A, maxit, shifted=False):
    """
    Given a symmetric matrix A, applies the modified QR algorithm
    (shifted or not shifted) and returns the result.

    :param A: an mxm-dimensional numpy array (must be symmetric)
    :param maxit: an integer, maximum number of iterations
    :param shifted: logical, if True applies the shifted
    QR algorithm

    :return e: an m-dimensional numpy array containing the
    eigenvalues of A
    :return off: a numpy array containing the values of |T_{k, k-1}|
    """

    m = len(A)
    hessenberg(A)

    e = []
    off = []

    for k in range(m-1, -1, -1):
        if k != 0:
            Ak, off_k = pure_QR(A, maxit, tridiagonal=True, shifted=shifted)
        else:
            Ak, off_k = pure_QR(A, maxit, tridiagonal=True, shifted=False)
        if k != 0:
            off.extend(off_k)
        e.append(Ak[k, k])
        A = Ak[:k, :k]

    return e, np.abs(off)

# Eigenvalues of A using the modified QR algoritm
A = createAij(5)
print(modified_QR(A, 5000)[0])

# Eigenvalues of A using the shifted modified QR algorithm
A = createAij(5)
print(modified_QR(A, 5000, shifted=True)[0])
