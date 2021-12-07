import numpy as np
from cla_utils import LU_inplace

def createA(n):
    """
    Given an integer n, creates a (4n+1)x(4n+1) matrix A of the form 
    given in the question.

    :input n: an integer

    :return A: a (4n+1)x(4n+1)-dimensional numpy array
    """

    A = np.zeros((4*n + 1, 4*n + 1))
    eps = np.random.uniform(0, 0.1)

    for i in range(n):
        A[4*i:4*i+5, 4*i:4*i+5] += np.reshape(eps * np.random.uniform(size=25), (5, 5))

    A += np.eye(4*n + 1)

    return A

A = createA(2)
LU = LU_inplace(A)
L = np.tril(LU, -1) + np.eye(len(A))
U = np.triu(LU)
print(L)
print(U)


def modified_LU(A):
    """
    Given a matrix A of the form given in the question, computes the
    LU decomposition (taking full advantage of its sparsity structure).

    :input: a (4n+1)x(4n+1)-dimensional numpy array

    :return A: a (4n+1)x(4n+1)-dimensional numpy array containing the
    values for reconstructing L and U.
    """

    m = len(A)

    for k in range(m-1):
        for j in range(k+1, min(m, k + 5 - k % 4)):
            A[j, k] /= A[k, k]
            n = min(m, k + 5 - k % 4)
            A[j, k+1:n] -= A[j, k] * A[k, k+1:n]
    
    return A
