import numpy as np


def evals(eps):
    """
    Given a real number epsilon, computes the eigenvalues of the
    perturbed identity matrix A_epsilon.

    :param eps: a real number

    :return: a 2-dimensional numpy array containing the eigenvalues
    """
    tr = 2 + eps
    det = 1 + eps

    delta = tr ** 2 - 4 * det

    l1 = (tr + np.sqrt(delta)) / 2
    l2 = (tr - np.sqrt(delta)) / 2

    return np.array([l1, l2])

eps = 10 ** -14

print(evals(0))
print(evals(eps))


true_lambda = np.array([1, 1 + eps])

print(np.linalg.norm(true_lambda - evals(eps)))
