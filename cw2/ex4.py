import numpy as np

def banded_LU(A, p, q):
    """
    Given a banded matrix A with bandwidths p and q, computes
    the LU factorization using the banded matrix algorithm (in-place).

    :input A: an mxm-dimensional numpy array
    :input p: an integer (lower bandwidth)
    :input q: an integer (upper bandwidth)

    :return op_cont: an integer representing the operation count of the algorithm
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


# matrix size
n = 9
delta = 1 / n
m = (n-1) ** 2

# parameters
mu, c, alpha = 1, 1, 1
s_0, r_0 = 1, 1

# grid of points
x = np.array([i * delta for i in range(n+1)])
y = np.array([i * delta for i in range(n+1)])

def b(x, y, alpha):
    """
    Computes b(x, y) given alpha.

    :input x: a real number
    :input y: a real number
    :input alpha: a real number

    :return b: a 2-dimensional list
    """
    return [alpha * (-np.sin(np.pi * x) * np.cos(np.pi * y)), alpha * (np.cos(np.pi * x) * np.sin(np.pi * y))]


def S(x, y, s_0, r_0):
    """
    Computes S(x, y) given s_0 and r_0.

    :input x: a real number
    :input y: a real number
    :input s_0: a real number
    :input r_0: a real number

    :return S: a real number
    """
    return s_0 * np.exp((-(x - 1/4) ** 2 - (y - 1/4) ** 2) / (r_0 ** 2))


def createA1(m):
    """
    Creates matrix A^1 (given its size m).

    :input m: an integer

    :return A1: an mxm-dimensional array
    """

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
    """
    Creates matrix A^2 (given its size m).

    :input m: an integer

    :return A2: an mxm-dimensional array
    """
    
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
    """
    Creates vector S1 used for computing vector S^1 (given its size n-1).

    :input n: an integer

    :return S1: an (n-1)^2-dimensional array
    """

    S1 = np.zeros((n-1, n-1))

    for i in range(n-1):
        for j in range(n-1):
            S1[i, j] = S(x[j+1], y[i+1], s_0, r_0)

    return S1.reshape((n-1) ** 2)


def createS2(n):
    """
    Creates vector S2 used for computing vector S^2 (given its size n-1).

    :input n: an integer

    :return S2: an (n-1)^2-dimensional array
    """

    S2 = np.zeros((n-1, n-1))

    for i in range(n-1):
        for j in range(n-1):
            S2[i, j] = S(x[i+1], y[j+1], s_0, r_0)
    
    return S2.reshape((n-1) ** 2)


def createB1(m):
    """
    Creates matrix B1 used for computing vector S^1 (given its size m).

    :input m: an integer

    :return B1: an mxm-dimensional array
    """

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
    """
    Creates matrix B2 used for computing vector S^2 (given its size m).

    :input m: an integer

    :return B2: an mxm-dimensional array
    """

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
    """
    Solves the differential equation using the iterative scheme, given
    the number of iterations, the initial guess v2 and the matrices required to
    construct the LHS and RHS of the equations.

    :input n_iter: an integer (number of iterations)
    :input v2: an m-dimensional numpy array
    :input A1: an mxm-dimensional array
    :input A2: an mxm-dimensional array
    :input B1: an mxm-dimensional array
    :input B2: an mxm-dimensional array
    :input S1: an (n-1)^2-dimensional array
    :input S2: an (n-1)^2-dimensional array

    :return v2: an m^2-dimensional array (containing the u-values)
    """

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
