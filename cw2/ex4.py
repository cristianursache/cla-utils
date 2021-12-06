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

