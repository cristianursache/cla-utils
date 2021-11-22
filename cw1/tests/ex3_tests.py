import pytest
from cw1.ex3 import householder_rv, invertQ_rv, solve_rv
from cla_utils import householder_qr, householder_ls
import numpy as np
import numpy.random as random

@pytest.mark.parametrize('m, n', [(10, 7), (25, 25), (60, 40), (150, 150)])
def test_R_householder_rv(m, n):
    random.seed(47)
    A = random.randn(m, n)

    Rv = householder_rv(A)
    R = householder_qr(A)[1]

    assert(np.linalg.norm(np.triu(Rv) - R) < 1.0e-06)

@pytest.mark.parametrize('m, n', [(10, 5), (30, 30), (70, 50), (140, 140)])
def test_invertQ_rv(m, n):
    random.seed(47)
    A = random.randn(m, n)
    b = random.rand(m)

    Rv = householder_rv(A)
    sol = invertQ_rv(Rv, b).T[0]
    Q = householder_qr(A)[0]

    assert(np.linalg.norm(sol - Q.T @ b) < 1.0e-06)

@pytest.mark.parametrize('m, n', [(10, 8), (20, 20), (55, 45), (170, 170)])
def test_solve_rv(m, n):
    random.seed(47)
    A = random.randn(m, n)
    b = random.randn(m)

    Rv = householder_rv(A)
    sol = solve_rv(Rv, b)

    assert(np.linalg.norm(sol - householder_ls(A, b)) < 1.0e-06)
