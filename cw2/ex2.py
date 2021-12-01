import numpy as np


def evals(eps):
    """
    
    """
    tr = 2 + eps
    det = 1 + eps

    delta = tr ** 2 - 4 * det

    l1 = (tr + np.sqrt(delta)) / 2
    l2 = (tr - np.sqrt(delta)) / 2

    return l1, l2

eps = 10 ** -14

print(evals(0))
print(evals(eps))
