import numpy as np

def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """

    u = np.zeros(np.shape(Q)[1])

    for j in range(np.shape(Q)[1]):
        u[j] = np.inner(np.conjugate(Q[:, j]), v)

    r = v - np.sum(np.inner(u[j], Q[:, j]) for j in range(len(u)))

    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """

    Q_star = np.conjugate(Q).T
    x = np.dot(Q_star, b)

    return x


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    P = np.dot(Q, np.conjugate(Q).T)

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mxl-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U, for appropriate l.
    """

    Q = np.linalg.qr(V, mode='complete')[0]
    Q_ort = Q[:, np.shape(V)[1]:]

    return Q_ort


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    
    R = np.zeros((np.shape(A)[1], np.shape(A)[1]))

    for j in range(np.shape(A)[1]):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(np.conjugate(A[:, i]), A[:, j])
            v -= R[i, j] * A[:, i]
        R[j, j] = np.linalg.norm(v)
        A[:, j] = v / R[j, j]

    return R

def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """

    R = np.zeros((np.shape(A)[1], np.shape(A)[1]))
    V = A.copy()

    for i in range(np.shape(A)[1]):
        R[i, i] = np.linalg.norm(V[:, i])
        A[:, i] = V[:, i] / R[i, i]
        for j in range(i+1, np.shape(A)[1]):
            R[i, j] = np.dot(np.conjugate(A[:, i]), V[:, j])
            V[:, j] -= R[i, j] * A[:, i]

    return R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is normalised and orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """

    R = np.eye(np.shape(A)[1])

    R[k, k] = np.linalg.norm(A[:, k])
    for j in range(k+1, np.shape(A)[1]):
        R[k, j] = np.dot(np.conjugate(A[:, k]), A[:, j])

    return R

def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:,:] = np.dot(A, Rk)
        R[:,:] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
