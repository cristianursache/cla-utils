import numpy as np
from cla_utils import householder_qr, pure_QR

def createA(n):
    A = np.zeros((2*n, 2*n))
    A += np.diag(np.ones(2*n - 1), 1)
    A -= np.diag(np.ones(2*n - 1), -1)
    return A

def ev_A(n, nit):
    A = createA(n)
    e = np.zeros(2 * n, dtype=complex)
    # tol does nothing in this case
    Ak = pure_QR(A, nit, .001)
    for i in range(n):
        e[i] = 1j *  Ak[2*i, 2*i + 1]
    e[n:] = -e[:n]
    return e

def createB(n):
    B = np.zeros((2*n, 2*n))
    B += 2 * np.diag(np.ones(2*n - 1), 1)
    B -= np.diag(np.ones(2*n - 1), -1)
    return B

def ev_B(n, nit):
    B = createB(n)
    e = np.zeros(2 * n, dtype=complex)
    # tol does nothing in this case
    Bk = pure_QR(B, nit, .001)
    for i in range(n):
        det = -Bk[2*i, 2*i + 1] * Bk[2*i + 1, 2*i]
        e[i] = 1j * np.sqrt(det)
    e[n:] = -e[:n]
    return e
