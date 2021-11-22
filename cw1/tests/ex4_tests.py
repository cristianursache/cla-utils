import pytest
from cw1.ex4 import solve, exp_search_int, binary_search
from numpy.linalg import norm
import numpy.random as random

@pytest.mark.parametrize('m, n', [(10, 7), (50, 30), (200, 150)])
def test_solve(m, n):
    random.seed(47)
    A = random.randn(m, n)
    b = random.randn(m)
    lbd = random.randn()

    x = solve(A, b, lbd)

    dif = A.T @ A @ x - A.T @ b + lbd * x

    assert(norm(dif) < 1.0e-09)

@pytest.mark.parametrize('m, n', [(10, 7), (50, 30), (200, 150)])
def test_binary_search(m, n):
    random.seed(47)
    A = random.randn(m, n)
    b = random.randn(m)
    
    maxlbd = exp_search_int(A, b)

    actual = binary_search(A, b, maxlbd)

    assert(abs(norm(solve(A, b, actual)) - 1) < 1.0e-06)
