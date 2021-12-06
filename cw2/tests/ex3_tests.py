import pytest
import numpy as np
from cw2.ex3 import createA, modified_LU


@pytest.mark.parametrize('n', [5, 10, 30])
def test_modified_LU(n):
    A = createA(n)
    A1 = 1.0 * A

    LU = modified_LU(A)
    L = np.tril(LU, -1) + np.eye(len(A))
    U = np.triu(LU)

    assert(np.linalg.norm(L @ U - A1) < 1e-06)
