# src/solvers.py
# Solvers con callback para contar iteraciones reales (SciPy >= 1.9 usa rtol/atol)

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.sparse.linalg import cg, gmres, LinearOperator

def _as_linear_op(M) -> Optional[LinearOperator]:
    if M is None:
        return None
    if isinstance(M, LinearOperator):
        return M
    # Si M es callable que resuelve M^{-1} y -> M(y)
    if callable(M):
        def matvec(x):
            return M(x)
        return LinearOperator(shape=None, matvec=matvec)  # shape la infiere SciPy
    return M  # ya es algo que SciPy sabe envolver

def solve_cg(A, b, rtol: float = 1e-8, maxiter: int = 2000, M=None) -> Tuple[np.ndarray, int, int]:
    """
    Conjugate Gradient para SPD: devuelve (x, iteraciones_reales, info)
    - info = 0 si SciPy declara convergencia; >0 si alcanzó maxiter.
    """
    it = {"k": 0}
    def cb(_xk):
        it["k"] += 1
    x, info = cg(A, b, rtol=rtol, atol=0.0, maxiter=maxiter, M=_as_linear_op(M), callback=cb)
    iters = it["k"] if info == 0 else info
    return x, iters, info

def solve_gmres(A, b, rtol: float = 1e-8, restart: int = 30, maxiter: int = 2000, M=None) -> Tuple[np.ndarray, int, int]:
    """
    GMRES(m=restart): devuelve (x, iteraciones_reales, info)
    - info = 0 si SciPy declara convergencia; >0 si alcanzó maxiter*restart (según criterio interno).
    Contamos callbacks para tener iteraciones reales aunque info=0.
    """
    it = {"k": 0}
    def cb(_rk):
        it["k"] += 1
   # x, info = gmres(A, b, rtol=rtol, atol=0.0, restart=restart, maxiter=maxiter, M=_as_linear_op(M), callback=cb)
    x, info = gmres(
        A, b,
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
        restart=restart,
        M=M,
        callback=cb,
        callback_type="legacy"  # <— agrega esto
    )

    iters = it["k"] if info == 0 else info
    return x, iters, info

def solve_cholesky(A, b) -> np.ndarray:
    """
    Cholesky denso (para SPD): usa numpy.linalg.cholesky.
    """
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x

def solve_qr_ls(A, b) -> np.ndarray:
    """
    Mínimos cuadrados densos vía QR (np.linalg.lstsq).
    """
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x
