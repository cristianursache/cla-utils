import numpy as np

def flat(x, order='C'):
    """
    Given a 2-dimensional array x, returns the equivalent
    serialised 1-dimensional array.

    :input u: an nxn-dimensional numpy array

    :return: an n^2-dimensional numpy array
    """

    return np.ndarray.flatten(x, order=order)

def expand(x, type):
    """
    Given a 1-dimensional array x, returns the equivalent
    unserialised 2-dimensional array.

    :input x: an m-dimensional numpy array
    :input type: string; if type == 'u', returns an nxn-dimensional array
    if type == 'x', returns an nx(n+1)-dimensional array
    if type == 'y', return an (n+1)xn-dimensional array

    :return: a rectangular numpy array of appropriate dimension
    """

    n = int(np.sqrt(len(x)))

    if type == 'u':
        return x.reshape(n, n)
    elif type == 'x':
        return x.reshape(n, n+1)
    elif type == 'y':
        return x.reshape(n+1, n)

def H_apply(v, l, mu):
    """
    Given a 1-dimensional array v, applies the transformation H = mu * I + lambda * A
    (for given mu and lambda).

    :input v: an m-dimensional numpy array
    :input l: float
    :input mu: float

    :return H: an m-dimensional numpy array (serialised H)
    """

    u = expand(v, type='u')
    n = len(u)

    Lx = 2 * u
    Lx[:n-1, :] -= u[1:, :]
    Lx[1:, :] -= u[:n-1, :]

    Ly = 2 * u
    Ly[:, :n-1] -= u[:, 1:]
    Ly[:, 1:] -= u[:, :n-1]

    L = Lx + Ly
    H = l * L + mu * u
    return flat(H)
