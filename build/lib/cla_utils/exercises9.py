import numpy as np
import numpy.random as random
from scipy.linalg import solve
from cla_utils.exercises3 import householder_qr

def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.
    
    :return A3: a 3x3 numpy array.
    """

    return np.array([[ 0.68557183+0.46550108j,  0.12934765-0.1622676j ,
                    0.24409518+0.25335939j],
                  [ 0.1531015 +0.66678983j,  0.45112492+0.18206976j,
                    -0.02633966+0.43477693j],
                  [-0.10817164-1.16879196j, -0.18446849+0.03755672j,
                   0.06430325-0.44757084j]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return np.array([[ 0.46870499+0.37541453j,  0.19115959-0.39233203j,
                    0.12830659+0.12102382j],
                  [ 0.90249603-0.09446345j,  0.51584055+0.84326503j,
                    -0.02582305+0.23259079j],
                  [ 0.75419973-0.52470311j, -0.59173739+0.48075322j,
                    0.51545446-0.21867957j]])


def pow_it(A, x0, tol, maxit, store_iterations=False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either 

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """
    
    x = x0 / np.linalg.norm(x0)
    if store_iterations:
        iter = [x]
    
    for _ in range(maxit):
        x = A @ x
        x /= np.linalg.norm(x)
        lambda0 = np.dot(x, A) @ x
        if store_iterations:
            iter.append(x)
        if np.linalg.norm(A @ x - lambda0 * x) < tol:
            break
    
    if store_iterations:
        return iter, lambda0
    else:
        return x, lambda0


def inverse_it(A, x0, mu, tol, maxit, store_iterations=False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """

    m = len(A)
    
    x = x0 / np.linalg.norm(x0)
    if store_iterations:
        iter_evec = [x]
        iter_eval = [np.dot(np.conjugate(x), A) @ x]

    for _ in range(maxit):
        x = solve(A - mu * np.eye(m), x)
        x /= np.linalg.norm(x)
        l = np.dot(np.conjugate(x), A) @ x
        if store_iterations:
            iter_evec.append(x)
            iter_eval.append(l)
        if np.linalg.norm(A @ x - l * x) < tol:
            break
    
    if store_iterations:
        return iter_evec, iter_eval
    else:
        return x, l


def rq_it(A, x0, tol, maxit, store_iterations=False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """

    m = len(A)
    
    x = x0 / np.linalg.norm(x0)
    l = np.dot(np.conjugate(x), A) @ x
    if store_iterations:
        iter_evec = [x]
        iter_eval = [l]
    
    for _ in range(maxit):
        x = solve(A - l * np.eye(m), x)
        x /= np.linalg.norm(x)
        l = np.dot(np.conjugate(x), A) @ x
        if store_iterations:
            iter_evec.append(x)
            iter_eval.append(l)
        if np.linalg.norm(A @ x - l * x) < tol:
            break

    if store_iterations:
        return iter_evec, iter_eval
    else:
        return x, l


def pure_QR(A, maxit, tol=1e-12, tridiagonal=False, shifted=False):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance

    :return Ak: the result
    """

    m = len(A)
    
    Ak = A.copy()

    if tridiagonal:
        off = []
    
    for _ in range(maxit):
        if shifted:
            a = Ak[m-1, m-1]
            b = Ak[m-1, m-2]
            delta = (Ak[m-2, m-2] - Ak[m-1, m-1]) / 2
            mu = a - np.sign(delta) * (b ** 2) / (np.abs(delta) + np.sqrt(delta ** 2 + b ** 2))
            Q, R = householder_qr(Ak - mu * np.eye(m))
            Ak = R @ Q
            Ak += mu * np.eye(m)
        else:
            Q, R = householder_qr(Ak)
            Ak = R @ Q
        
        if tridiagonal:
            off.append(Ak[m-1, m-2])
            if np.abs(Ak[m-1, m-2]) < tol:
                break
        else:
            T = np.tril(Ak, -1)
            if np.linalg.norm(T) < tol:
                break
    
    if tridiagonal:
        return Ak, off
    else:
        return Ak
