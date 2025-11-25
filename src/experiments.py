# -----------------------------------------------------------
# experiments.py (VERSIÓN EXTENDIDA CON MODO AUTOMÁTICO)
# -----------------------------------------------------------

from __future__ import annotations
import argparse, csv, os, time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from typing import Optional

from problems import laplacian_1d, dense_spd, dense_ls
from solvers import solve_cg, solve_gmres, solve_cholesky, solve_qr_ls


# ===========================================================
# PRECONDICIONADORES
# ===========================================================

def make_precond_ilu(A: sp.csr_matrix) -> Optional[LinearOperator]:
    from scipy.sparse.linalg import spilu
    ilu = spilu(A.tocsc())

    def Mv(x):
        return ilu.solve(x)

    return LinearOperator(A.shape, matvec=Mv)


def make_precond_amg(A: sp.csr_matrix) -> Optional[LinearOperator]:
    try:
        import pyamg
    except Exception:
        return None

    ml = pyamg.ruge_stuben_solver(A)

    def Mv(x):
        return ml.solve(x, tol=0.0, maxiter=1)

    return LinearOperator(A.shape, matvec=Mv)


# ===========================================================
# UTILIDADES
# ===========================================================

def run_and_time(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def write_header_if_needed(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        ensure_parent(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Caso",
                "n",
                "Tipo",
                "Método",
                "Prec.",
                "Iter.",
                "||r||2",
                "Tiempo (s)",
                "tol",
                "maxiter",
                "restart",
            ])


def append_row(path: str, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)


def build_problem(case: str, n: int):
    case = case.lower()
    if case == "lap1d":
        A, b = laplacian_1d(n)
        return "LAP1D", "DISPERSA", A, b

    elif case == "dense_spd":
        A, b = dense_spd(n)
        return "DENSE_SPD", "DENSA SPD", A, b

    elif case == "dense_ls":
        A, b = dense_ls(2 * n, n)
        return "DENSE_LS", "DENSA LS", A, b

    else:
        raise ValueError(f"Case no reconocido: {case}")


# ===========================================================
# NUEVA FUNCIÓN: MODO AUTOMÁTICO "RUN ALL"
# ===========================================================


def run_all_experiments():
    """
    Corre baterias de pruebas para:
    - LAP1D      (dispersa)
    - DENSE_SPD  (densa SPD)
    - DENSE_LS   (minimos cuadrados, matriz rectangular)

    Resultados se guardan en: report/resultados_completos.csv
    """

    OUTPUT = "report/resultados_completos.csv"
    write_header_if_needed(OUTPUT)

    # Definimos por caso que metodos y n vamos a usar
    casos_config = [
        # caso,      lista de n,                 metodos permitidos,                 usar_precond
        ("lap1d",    [500, 1000, 2000],         ["CG", "GMRES", "CHOLESKY", "QR"], True),
        ("dense_spd",[500, 1000, 2000],         ["CG", "GMRES", "CHOLESKY", "QR"], False),
        ("dense_ls", [500, 1000],               ["QR"],                              False),  # SOLO QR, A es rectangular
    ]

    for case_name, n_values, methods, allow_prec in casos_config:
        for n in n_values:
            caso_nom, tipo, A, b = build_problem(case_name, n)

            # Precondicionadores:
            # - Para LAP1D (dispersa) probamos none / ILU / AMG
            # - Para los demas solo sin precondicionador
            if allow_prec and sp.issparse(A):
                preconds = ["none", "ilu", "amg"]
            else:
                preconds = ["none"]

            for method in methods:
                for prec in preconds:

                    # Métodos directos no usan precondicionador distinto de none
                    if method in ("CHOLESKY", "QR") and prec != "none":
                        continue

                    # --- Precondicionador ---
                    prec_name = "—"
                    M = None
                    if prec != "none" and sp.issparse(A):
                        if prec == "ilu":
                            M = make_precond_ilu(A)
                            prec_name = "ILU"
                        elif prec == "amg":
                            M = make_precond_amg(A)
                            prec_name = "AMG" if M is not None else "—"

                    # --- Ejecutar método ---
                    if method == "CG":
                        (x, it, info), t = run_and_time(
                            solve_cg, A, b,
                            rtol=1e-8, maxiter=2000, M=M
                        )

                    elif method == "GMRES":
                        (x, it, info), t = run_and_time(
                            solve_gmres, A, b,
                            rtol=1e-8, maxiter=2000, restart=30, M=M
                        )

                    elif method == "CHOLESKY":
                        A2 = A.toarray() if sp.issparse(A) else A
                        x, t = run_and_time(solve_cholesky, A2, b)
                        it, info = None, None

                    elif method == "QR":
                        A2 = A.toarray() if sp.issparse(A) else A
                        x, t = run_and_time(solve_qr_ls, A2, b)
                        it, info = None, None

                    # --- Residuos y escritura ---
                    r = (A @ x) - b if sp.issparse(A) else (A @ x - b)
                    res2 = float(np.linalg.norm(r))

                    append_row(OUTPUT, [
                        caso_nom, n, tipo,
                        method,
                        prec_name,
                        "" if it is None else it,
                        f"{res2:.16g}",
                        t,
                        1e-8,
                        "" if method in ("CHOLESKY", "QR") else 2000,
                        "" if method != "GMRES" else 30,
                    ])

                    print(f"[OK] {caso_nom} n={n} metodo={method} prec={prec_name}")

# def run_all_experiments():
#     OUTPUT = "resultados_completos.csv"
#     write_header_if_needed(OUTPUT)
#
#     cases = ["lap1d", "dense_spd", "dense_ls"]
#     methods = ["CG", "GMRES", "CHOLESKY", "QR"]
#     preconds = ["none", "ilu", "amg"]
#
#     n_values = [500, 1000, 2000]
#
#     for case in cases:
#         for n in n_values:
#             caso_nom, tipo, A, b = build_problem(case, n)
#
#             for method in methods:
#                 for prec in preconds:
#
#                     # Métodos directos NO necesitan precondicionadores
#                     if method in ("CHOLESKY", "QR") and prec != "none":
#                         continue
#
#                     # Precondicionador
#                     prec_name = "—"
#                     M = None
#
#                     if prec != "none" and sp.issparse(A):
#                         if prec == "ilu":
#                             M = make_precond_ilu(A)
#                             prec_name = "ILU"
#                         elif prec == "amg":
#                             M = make_precond_amg(A)
#                             prec_name = "AMG" if M else "—"
#
#                     # Ejecutar método
#                     if method == "CG":
#                         (x, it, info), t = run_and_time(
#                             solve_cg, A, b,
#                             rtol=1e-8, maxiter=2000, M=M
#                         )
#
#                     elif method == "GMRES":
#                         (x, it, info), t = run_and_time(
#                             solve_gmres, A, b,
#                             rtol=1e-8, maxiter=2000, restart=30, M=M
#                         )
#
#                     elif method == "CHOLESKY":
#                         A2 = A.toarray() if sp.issparse(A) else A
#                         x, t = run_and_time(solve_cholesky, A2, b)
#                         it, info = None, None
#
#                     elif method == "QR":
#                         A2 = A.toarray() if sp.issparse(A) else A
#                         x, t = run_and_time(solve_qr_ls, A2, b)
#                         it, info = None, None
#
#                     # Residuos
#                     r = (A @ x) - b if sp.issparse(A) else (A @ x - b)
#                     res2 = float(np.linalg.norm(r))
#
#                     # Escribir fila en CSV
#                     append_row(OUTPUT, [
#                         caso_nom, n, tipo,
#                         method,
#                         prec_name,
#                         "" if it is None else it,
#                         f"{res2:.16g}",
#                         t,
#                         1e-8,
#                         "" if method in ("CHOLESKY", "QR") else 2000,
#                         "" if method != "GMRES" else 30
#                     ])
#
#                     print(f"[OK] {caso_nom} n={n} método={method} prec={prec_name}")


# ===========================================================
# MAIN NORMAL Y AUTOMÁTICO
# ===========================================================

def main():

    p = argparse.ArgumentParser()

    p.add_argument("--auto", action="store_true",
                   help="Ejecuta TODOS los casos automáticamente.")

    # Modo clásico
    p.add_argument("--case", choices=["lap1d", "dense_spd", "dense_ls"])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--method", choices=["CG", "GMRES", "CHOLESKY", "QR"])
    p.add_argument("--precond", choices=["none", "ilu", "amg"], default="none")
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--restart", type=int, default=30)
    p.add_argument("--out", type=str)
    p.add_argument("--append", action="store_true")

    args = p.parse_args()

    # ◼️ MODO AUTOMÁTICO
    if args.auto:
        print(">>> EJECUTANDO TODAS LAS CORRIDAS (AUTO MODE) <<<")
        run_all_experiments()
        print(">>> COMPLETADO. Resultados en: resultados_completos.csv")
        return

    # ◼️ MODO NORMAL (no se modifica)
    if not args.out:
        raise ValueError("Debe usar --out archivo.csv")

    caso_nom, tipo, A, b = build_problem(args.case, args.n)

    # PRECOND
    prec_name = "—"
    M = None
    if args.precond != "none" and sp.issparse(A):
        if args.precond == "ilu":
            M = make_precond_ilu(A)
            prec_name = "ILU"
        elif args.precond == "amg":
            M = make_precond_amg(A)
            prec_name = "AMG" if M else "—"

    # MÉTODOS
    method = args.method

    if method == "CG":
        (x, it, info), t = run_and_time(
            solve_cg, A, b,
            rtol=args.tol, maxiter=args.maxiter, M=M
        )
    elif method == "GMRES":
        (x, it, info), t = run_and_time(
            solve_gmres, A, b,
            rtol=args.tol, maxiter=args.maxiter,
            restart=args.restart, M=M
        )
    elif method == "CHOLESKY":
        A2 = A.toarray() if sp.issparse(A) else A
        x, t = run_and_time(solve_cholesky, A2, b)
        it, info = None, None
    elif method == "QR":
        A2 = A.toarray() if sp.issparse(A) else A
        x, t = run_and_time(solve_qr_ls, A2, b)
        it, info = None, None

    # RESIDUO
    r = (A @ x) - b if sp.issparse(A) else (A @ x - b)
    res2 = float(np.linalg.norm(r))

    # CSV
    write_header_if_needed(args.out)
    append_row(args.out, [
        caso_nom, args.n, tipo,
        method,
        prec_name,
        "" if it is None else it,
        f"{res2:.16g}",
        t,
        args.tol,
        "" if args.method in ("CHOLESKY", "QR") else args.maxiter,
        "" if args.method != "GMRES" else args.restart,
    ])

    print("Fila escrita.")


if __name__ == "__main__":
    main()
