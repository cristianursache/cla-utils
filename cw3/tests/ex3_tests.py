from numpy.lib.polynomial import polyder
import pytest
import numpy as np
from numpy import random
from cw3.ex3 import flat, expand, H_apply

@pytest.mark.parametrize('n', [5, 10, 15, 30])
def test_flat_expand(n):
    A = random.randn(n, n)
    v = flat(A)
    u = expand(v, type='u')
    assert np.linalg.norm(A - u) == 0

@pytest.mark.parametrize('n', [5, 10, 15, 30])
def test_H_apply(n):
    v = np.ones(n ** 2)
    L = H_apply(v, l=1, mu=0)
    L_mat = expand(L, type='u')
    n = len(L_mat)
    assert np.linalg.norm(L_mat[1:n-1, 1:n-1]) == 0
