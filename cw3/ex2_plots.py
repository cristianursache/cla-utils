import numpy as np
from numpy import random
from ex2 import createAij, modified_QR
from cla_utils import pure_QR
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Random 10x10 matrix
R = random.randn(10, 10)
A = R + R.T
off = modified_QR(A, 5000)[1]
# Modified QR number of iterations
print(len(off))

# Modified QR plot for random matrix
plt.figure()
plt.plot(off, linewidth=2)
plt.yscale('log')
plt.xlabel('Iteration number')
plt.ylabel(r'$|T_{k, k-1}|$')
plt.title('Modified QR for random matrix')
plt.show()

R = random.randn(10, 10)
A = R + R.T
# Pure QR number of iterations
print(pure_QR(A, 5000, count_its=True))

# Matrix A
A = createAij(5)
off = modified_QR(A, 5000)[1]
# Modified QR number of iterations
print(len(off))

# Modified QR plot for matrix A
plt.figure()
plt.plot(off, linewidth=2)
plt.yscale('log')
plt.xlabel('Iteration number')
plt.ylabel(r'$|T_{k, k-1}|$')
plt.title('Modified QR for matrix 'r'$A$')
plt.show()

A = createAij(5)
# Pure QR number of iterations
print(pure_QR(A, 5000, count_its=True))

A = createAij(5)
off = modified_QR(A, 5000, shifted=True)[1]

# Modified QR (shifted) plot for matrix A
plt.figure()
plt.plot(off, linewidth=2)
plt.yscale('log')
plt.xlabel('Iteration number')
plt.ylabel(r'$|T_{k, k-1}|$')
plt.title('Modified QR (shifted) for matrix 'r'$A$')
plt.show()

D = np.diag([i for i in range(15, 0, -1)], 0)
O = np.ones((15, 15))
A = D + O
off = modified_QR(A, 5000)[1]

# Modified QR for matrix D + O
plt.figure()
plt.plot(off, linewidth=2)
plt.yscale('log')
plt.xlabel('Iteration number')
plt.ylabel(r'$|T_{k, k-1}|$')
plt.title('Modified QR for matrix 'r'D + O')
plt.show()

D = np.diag([i for i in range(15, 0, -1)], 0)
O = np.ones((15, 15))
A = D + O
off = modified_QR(A, 5000, shifted=True)[1]

# Modified QR (shifted) for matrix D + O
plt.figure()
plt.plot(off, linewidth=2)
plt.yscale('log')
plt.xlabel('Iteration number')
plt.ylabel(r'$|T_{k, k-1}|$')
plt.title('Modified QR (shifted) for matrix 'r'D + O')
plt.show()
