import numpy as np

def banded_LU(A, p, q):
    """
    
    """

    m = len(A)

    op_cnt = 0

    for k in range(m-1):
        for j in range(k+1, min(m, k+p+1)):
            A[j, k] /= A[k, k]
            n = min(m, k+q+1)
            A[j, k+1:n] -= A[j, k] * A[k, k+1:n]
            op_cnt += q
    
    return op_cnt


n = 10
delta = 1 / n
m = (n-1) ** 2

mu, c, alpha = 1, 1, 1
s_0, r_0 = 1, 1

x = np.array([i * delta for i in range(n+1)])
y = np.array([i * delta for i in range(n+1)])

def b(x, y, alpha):
    return [alpha * (-np.sin(np.pi * x) * np.cos(np.pi * y)), alpha * (np.cos(np.pi * x) * np.sin(np.pi * y))]


def S(x, y, s_0, r_0):
    return s_0 * np.exp((-(x - 1/4) ** 2 - (y - 1/4) ** 2) / (r_0 ** 2))


def createA1(m):

    A1 = np.zeros((m, m))

    for k in range(m):
        if k != 0:
            i, j = int(k % np.sqrt(m) + 1), int(k // np.sqrt(m) + 1)
            val = -b(x[i], y[j], alpha)[0] / (2 * delta) - mu / (delta ** 2)
            A1[k, k-1] = val
        if k != m-1:
            i, j = int(k % np.sqrt(m) + 1), int(k // np.sqrt(m) + 1)
            val = b(x[i], y[j], alpha)[0] / (2 * delta) - mu / (delta ** 2)
            A1[k, k+1] = val
        A1[k, k] = 4 * mu / (delta ** 2) + c
    
    return A1


def createA2(m):
    
    A2 = np.zeros((m, m))

    for k in range(m):
        if k != 0:
            i, j = int(k // np.sqrt(m) + 1), int(k % np.sqrt(m) + 1)
            val = -b(x[i], y[j], alpha)[1] / (2 * delta) - mu / (delta ** 2)
            A2[k, k-1] = val
        if k != m-1:
            i, j = int(k % np.sqrt(m) + 1), int(k // np.sqrt(m) + 1)
            val = b(x[i], y[j], alpha)[1] / (2 * delta) - mu / (delta ** 2)
            A2[k, k+1] = val
        A2[k, k] = 4 * mu / (delta ** 2) + c
    
    return A2


def createS1(n):

    S1 = np.zeros((n-1, n-1))

    for i in range(n-1):
        for j in range(n-1):
            S1[i, j] = S(x[j+1], y[i+1], s_0, r_0)

    return S1.reshape((n-1) ** 2)


def createS2(n):

    S2 = np.zeros((n-1, n-1))

    for i in range(n-1):
        for j in range(n-1):
            S2[i, j] = S(x[i+1], y[j+1], s_0, r_0)
    
    return S2.reshape((n-1) ** 2)


def createB1(m):

    B1 = np.zeros((m, m))

    for k in range(m):
        if k != 0:
            i, j = int(k % np.sqrt(m) + 1), int(k // np.sqrt(m) + 1)
            val = b(x[i], y[j], alpha)[1] / (2 * delta) + mu / (delta ** 2)
            B1[k, k-1] = val
        if k != m-1:
            i, j = int(k % np.sqrt(m) + 1), int(k // np.sqrt(m) + 1)
            val = -b(x[i], y[j], alpha)[1] / (2 * delta) + mu / (delta ** 2)
            B1[k, k+1] = val
    
    return B1


def createB2(m):

    B2 = np.zeros((m, m))

    for k in range(m):
        if k != 0:
            i, j = int(k // np.sqrt(m) + 1), int(k % np.sqrt(m) + 1)
            val = b(x[i], y[j], alpha)[0] / (2 * delta) + mu / (delta ** 2)
            B2[k, k-1] = val
        if k != m-1:
            i, j = int(k % np.sqrt(m) + 1), int(k // np.sqrt(m) + 1)
            val = -b(x[i], y[j], alpha)[0] / (2 * delta) + mu / (delta ** 2)
            B2[k, k+1] = val
    
    return B2


def solve(n_iter, v2, A1, A2, B1, B2, S1, S2):

    for i in range(n_iter):
        RHS1 = S1 + B1 @ v2
        v1 = np.linalg.solve(A1, RHS1)
        RHS2 = S2 + B2 @ v1
        v2 = np.linalg.solve(A2, RHS2)

    return v2


v2 = np.random.randn((n-1) ** 2)
A1, A2 = createA1(m), createA2(m)
B1, B2 = createB1(m), createB2(m)
S1, S2 = createS1(n), createS2(n)

u = solve(500, v2, A1, A2, B1, B2, S1, S2).reshape((n-1, n-1))
print(u)
