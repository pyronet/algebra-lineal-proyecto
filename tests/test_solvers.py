# tests/test_solvers.py
import numpy as np
import scipy.sparse as sp

from src.problems import laplacian_1d, dense_spd
from src.solvers import solve_cg, solve_cholesky

def test_cg_lap1d_small():
    A, b = laplacian_1d(50)
    x, iters, info = solve_cg(A, b, rtol=1e-8, maxiter=2000, M=None)
    r = A @ x - b
    assert np.linalg.norm(r) < 1e-6

def test_cholesky_dense_spd():
    A, b = dense_spd(50)
    x = solve_cholesky(A, b)
    r = A @ x - b
    assert np.linalg.norm(r) < 1e-10
