import pytest
from numpy.linalg import norm
from cw1.ex1 import C, compress

@pytest.mark.parametrize('C', [C])
def test_compression(C):
    assert norm(C - compress(C)) < 1.0e-06
