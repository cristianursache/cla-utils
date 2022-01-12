import numpy as np
from cla_utils import hessenberg, pure_QR

def createAij(n):
    i, j = np.indices((n, n))
    A = 1 / (1 + i + j)
    return A

A = createAij(5)
print(np.linalg.eig(A)[0])

hessenberg(A)
Ak = pure_QR(A, 10000, tridiagonal=True)
print(np.diag(Ak, 0))
