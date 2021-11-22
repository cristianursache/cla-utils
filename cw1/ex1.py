from cla_utils import *
import numpy as np

C = np.loadtxt('C.dat', delimiter=',')

# compute the QR decomposition of C
Q, R = householder_qr(C)

def compress(C):
    """
    Given the matrix C, compress it based upon its QR factorization.

    :param C: 1000x100 data matrix C

    :return C_comp: compressed version of C
    """

    Q, R = householder_qr(C)
    C_comp = np.dot(Q[:, :3], R[:3, :])
    return C_comp

# compute the difference between compressed and original
dif = C - compress(C)

print("Norm of the difference is: {}".format(np.linalg.norm(dif)))
