import numpy as np


def get_Lk(m, lvec):
    """Compute the lower triangular row operation mxm matrix L_k 
    which has ones on the diagonal, and below diagonal entries
    in column k given by lvec (k is inferred from the size of lvec).

    :param m: integer giving the dimensions of L.
    :param lvec: a m-k-1 dimensional numpy array.

    :return Lk: an mxm dimensional numpy array.

    """

    k = m - len(lvec) - 1
    
    Lk = np.eye(m)
    Lk[k+1:, k] = -lvec

    return Lk



def LU_inplace(A):
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array

    """

    m = len(A)                 

    # 2 for loops implementation:
    #for k in range(m-1):
        #for j in range(k+1, m):
            #A[j, k] /= A[k, k]
            #A[j, k+1:] -= A[j, k] * A[k, k+1:]
    
    # 1 for loop & outer product implementation:
    for k in range(m-1):
        lk = np.zeros(m)
        lk[k+1:] = -A[k+1:, k] / A[k, k]
        ek = np.zeros(m)
        ek[k] = 1
        outer = np.outer(lk, ek)
        A[k:, k:] += outer[k:, k:] @ A[k:, k:]
        A -= outer

    return A


def solve_L(L, b):
    """
    Solve systems Lx_i=b_i for x_i with L lower triangular, i=1,2,...,k

    :param L: an mxm-dimensional numpy array, assumed lower triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """

    m, k = np.shape(b)

    x = np.zeros((m, k))

    for i in range(k):
        x[0, i] = b[0, i] / L[0, 0]
        for j in range(1, m):
            x[j, i] = (b[j, i] - np.dot(L[j, :j], x[:j, i])) / L[j, j]

    return x


def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing 
       b_i
    :return x: an mxk-dimensional numpy array, with ith column containing 
       the solution x_i

    """
                     
    m, k = np.shape(b)

    x = np.zeros((m, k))

    for i in range(k):
        x[-1, i] = b[-1, i] / U[-1, -1]
        for j in range(m-2, -1, -1):
            x[j, i] = (b[j, i] - np.dot(U[j, j+1:], x[j+1:, i])) / U[j, j]

    return x


def inverse_LU(A):
    """
    Form the inverse of A via LU factorisation.

    :param A: an mxm-dimensional numpy array.

    :return X: an mxm-dimensional numpy array.

    """

    m = len(A)                 
    
    LU = LU_inplace(A)
    L = np.tril(LU, -1) + np.eye(m)
    U = np.triu(LU)
    
    Y = solve_L(L, np.eye(m))
    X = solve_U(U, Y)

    return X
