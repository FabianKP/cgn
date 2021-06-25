# Proof that NNLS is faster than FNNLS

from scipy.optimize import nnls as scipy_nnls
from fnnls import fnnls

import numpy as np

from time import time

# make random test problem
m = 1000
n = 3000
A = np.random.randn(m, n)
# make random measurement
x = np.random.randn(n).clip(min=0.)
b = A @ x + 0.1 * np.random.randn(m)

# solve with nnls
print("Solving with nnls")
t0 = time()
x_nnls, r_nnls = scipy_nnls(A, b)
t_nnls = time() - t0

print("Solving with fast nnls")
# solve with fnnls
t1 = time()
x_fnnls, r_fnnls = fnnls(A, b)
t_fnnls = time() - t1

print(f"NNLS time: {t_nnls}")
print(f"FNNLS time: {t_fnnls}")

print(f"r_nnls = {r_nnls}")
print(f"r_fnnls = {r_fnnls}")