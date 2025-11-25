# src/datasets.py
import numpy as np
from scipy.sparse import diags, csc_matrix

def spd_random(n, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n,n))
    A = M.T @ M + 1e-6*np.eye(n)
    b = rng.standard_normal(n)
    return A, b

def laplacian_1d(n):
    main = 2*np.ones(n); off = -1*np.ones(n-1)
    A = diags([off, main, off], [-1,0,1], shape=(n,n), format="csc")
    return A

def rhs(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n)
