import pytest
import numpy as np
from cw3.ex2 import createAij
from cla_utils import hessenberg, pure_QR

@pytest.mark.parametrize('n', [5])
def test_mod_QR(n):
    A = createAij(n)
    e = np.linalg.eig(A)[0]
    hessenberg(A)
    Ak = pure_QR(A, 10000, tridiagonal=True)
    e_QR = np.diag(Ak, 0)
    assert np.linalg.norm(e - e_QR) < 1e-06
