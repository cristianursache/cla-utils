import pytest
import numpy as np
from cw3.ex1 import createA, ev_A, createB, ev_B

@pytest.mark.parametrize('n', [5, 10, 30])
def test_ev_A(n):
    A = createA(n)
    e = ev_A(n, 10000)
    for eval in e:
        assert np.abs(np.linalg.det(A - eval * np.eye(2*n))) < 1e-06

@pytest.mark.parametrize('n', [5, 10, 30])
def test_ev_B(n):
    B = createB(n)
    e = ev_B(n, 10000)
    for eval in e:
        assert np.abs(np.linalg.det(B - eval * np.eye(2*n))) < 1e-06
