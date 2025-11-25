# src/problems.py
# Generadores de matrices de prueba (dispersas y densas)

from __future__ import annotations
from typing import Tuple
import numpy as np
import scipy.sparse as sp

def laplacian_1d(n: int) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Laplaciano 1D (tridiagonal SPD) tamaño n, con b=ones.
    """
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    A = sp.diags([off, main, off], offsets=[-1, 0, 1], shape=(n, n), format="csr")
    b = np.ones(n)
    return A, b

def dense_spd(n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matriz densa SPD via A = M^T M + alpha I, b = ones.
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    A = M.T @ M + 1e-3 * np.eye(n)
    b = np.ones(n)
    return A, b

def dense_ls(m: int, n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Problema de mínimos cuadrados: A (m x n), b (m), devolver para resolver min ||Ax - b||.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    b = A @ x_true + 1e-3 * rng.standard_normal(m)
    return A, b
