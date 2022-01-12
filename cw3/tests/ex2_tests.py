import pytest
import numpy as np
from numpy import random
from cw3.ex2 import createAij, modified_QR
from cla_utils import hessenberg, pure_QR

@pytest.mark.parametrize('n', [5])
def test_pure_QR_tridiag(n):
    A = createAij(n)
    A0 = 1.0 * A
    hessenberg(A)
    Ak = pure_QR(A, 10000, tridiagonal=True)[0]
    e = np.diag(Ak, 0)
    for eval in e:
        assert np.abs(np.linalg.det(A0 - eval * np.eye(n))) < 1e-06

@pytest.mark.parametrize('n', [5, 7, 10])
def test_modified_QR(n):
    R = random.randn(n, n)
    A = R + R.T
    A0 = 1.0 * A
    e = modified_QR(A, 10000)[0]
    for eval in e:
        assert np.abs(np.linalg.det(A0 - eval * np.eye(n))) < 1e-04
