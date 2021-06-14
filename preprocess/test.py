from numba import jit
import numba
import numpy as np
import time


@jit("f8[:, :](f8[:, :])", nopython=True)
def fast_transpose(u):
    return np.transpose(u)

@jit("f8[:](f8[:, :])", nopython=True)
def fast_diag(u):
    return np.diag(u)

@jit("f8[:, :](f8[:, :], f8[:, :])", nopython=True)
def fast_dot(u, v):
    return np.tensordot(u, v)

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
for _ in range(1000):
    x = np.arange(100).reshape(20, 5).astype(np.float64)
    res2 = np.transpose(x)
    res = np.diag(x)
    res = np.dot(x, res2)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
for _ in range(1000):
    x = np.ones((20, 5), dtype=np.float64)
    x2 = np.ones((5, 20), dtype=np.float64)

    res2 = fast_transpose(x)
    res = fast_diag(x)
    res = fast_dot(x, x2)

end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))