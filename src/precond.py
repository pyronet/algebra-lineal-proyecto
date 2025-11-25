# src/precond.py
from scipy.sparse.linalg import spilu, LinearOperator
def ilu_precond(A, drop_tol=1e-3):
    ilu = spilu(A, drop_tol=drop_tol)
    return LinearOperator(A.shape, matvec=lambda v: ilu.solve(v))

def amg_precond(A):
    import pyamg
    ml = pyamg.smoothed_aggregation_solver(A)
    return ml.aspreconditioner()
