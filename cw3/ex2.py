import numpy as np
from cla_utils import hessenberg, pure_QR
import matplotlib.pyplot as plt

def createAij(n):
    i, j = np.indices((n, n))
    A = 1 / (1 + i + j)
    return A

A = createAij(5)
print(np.linalg.eig(A)[0])

hessenberg(A)
Ak = pure_QR(A, 10000, tridiagonal=True)[0]
print(np.diag(Ak, 0))

def modified_QR(A, maxit):
    e = []
    off = []
    m = len(A)
    hessenberg(A)
    for k in range(m-1, -1, -1):
        Ak, off_k = pure_QR(A, maxit, tridiagonal=True)
        off.extend(off_k)
        e.append(Ak[k, k])
        A = Ak[:k, :k]
    return e, off

A = createAij(5)
print(modified_QR(A, 5000)[0])

A = createAij(5)
plt.figure()
plt.plot(modified_QR(A, 5000)[1])
plt.yscale('log')
