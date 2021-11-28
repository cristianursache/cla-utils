import numpy as np


def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.

    :param A: an mxn-dimensional numpy array

    :return o2norm: the norm
    """

    s = max(np.linalg.eig(A.T @ A)[0])
    o2norm = np.sqrt(s)

    return o2norm


def test_ineq_1(A, x):

    return (np.linalg.norm(A @ x) <= operator_2_norm(A) * np.linalg.norm(x))


def test_ineq_2(A, B):

    return (operator_2_norm(A @ B) <= operator_2_norm(A) * operator_2_norm(B))


def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :return A: an mxn-dimensional numpy array

    :param ncond: the condition number
    """

    raise NotImplementedError

    return ncond
