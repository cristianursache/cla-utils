from cla_utils import *
import numpy as np
from scipy.linalg import solve_triangular

# create array of x's
x = np.arange(0., 1.0001, 1./51)

# create array of f's
f = np.zeros(len(x))
f[0], f[-2] = 1, 1

def vandermonde(x, deg=13):
    """
    Given values x, creates the Vandermonde matrix corresponding to the interpolation 
    polynomial of a given degree.

    :param x: array containing the values that interpolate the polynomial
    :param deg: degree of the interpolating polynomial

    :return V: Vandermonde matrix of the interpolating polynomial
    """

    V = np.zeros((len(x), deg))
    for i in range(np.shape(V)[0]):
        for j in range(np.shape(V)[1]):
            V[i, j] = x[i] ** j
    
    return V

def cGS_interpolation(V, f):
    """
    Given a Vandermonde matrix V and values f, finds the coefficients of the
    interpolating polynomial using classical Gram-Schmidt QR decomposition.

    :param V: Vandermonde matrix
    :param f: array containing the values such that p(x_j) = f_j

    :return alpha: array of coefficients of the interpolating polynomial
    """

    Q = V.copy()
    R = GS_classical(Q)
    alpha = solve_triangular(R, np.dot(Q.T, f))
    return alpha

def mGS_interpolation(V, f):
    """
    Given a Vandermonde matrix V and values f, finds the coefficients of the
    interpolating polynomial using modified Gram-Schmidt QR decomposition.

    :param V: Vandermonde matrix
    :param f: array containing the values such that p(x_j) = f_j

    :return alpha: array of coefficients of the interpolating polynomial
    """

    Q = V.copy()
    R = GS_modified(Q)
    alpha = solve_triangular(R, np.dot(Q.T, f))
    return alpha

def householder_interpolation(V, f):
    """
    Given a Vandermonde matrix V and values f, finds the coefficients of the
    interpolating polynomial using Householder QR decomposition.

    :param V: Vandermonde matrix
    :param f: array containing the values such that p(x_j) = f_j

    :return alpha: array of coefficients of the interpolating polynomial
    """

    alpha = householder_ls(V, f)
    return alpha
