import pytest
import numpy as np
from numpy import random
from cw2.ex4 import banded_LU

@pytest.mark.parametrize('n, p, q', [(10, 2, 2), (50, 5, 4), (100, 8, 10)])
def test_banded_LU(n, p, q):
    A = np.eye(n)
    for i in range(1, p+1):
        A += np.diag(random.randn(n-i), -i)
    for i in range(1, q+1):
        A+= np.diag(random.randn(n-i), i)

    A1 = 1.0 * A

    banded_LU(A, p, q)
    L = np.tril(A, -1) + np.eye(len(A))
    U = np.triu(A)

    assert(np.linalg.norm(L @ U - A1) < 1e-06)
