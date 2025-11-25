import os, subprocess, sys

OUT = r"C:/Users/Administrator/IdeaProjects/algebra-lineal-proyecto/report/tabla_resultados.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CASES = []

# LAP1D: CG CON Y SIN PRECOND
for n in [1000, 2000, 5000]:
    for pre in ["none", "ilu", "amg"]:
        CASES.append(["--case","lap1d","--n",str(n),"--method","CG","--precond",pre,"--tol","1e-8","--maxiter","4000"])

# LAP1D: GMRES CON Y SIN PRECOND
for n in [2000, 5000]:
    for pre in ["none", "ilu", "amg"]:
        CASES.append(["--case","lap1d","--n",str(n),"--method","GMRES","--restart","30","--precond",pre,"--tol","1e-8","--maxiter","4000"])

# DENSE_SPD: CHOLESKY
for n in [500, 1000]:
    CASES.append(["--case","dense_spd","--n",str(n),"--method","CHOLESKY"])

# DENSE_SPD: CG SIN PRECOND
for n in [500, 1000]:
    CASES.append(["--case","dense_spd","--n",str(n),"--method","CG","--precond","none","--tol","1e-10","--maxiter","4000"])

# DENSE_LS: QR
for n in [500, 1000]:
    CASES.append(["--case","dense_ls","--n",str(n),"--method","QR"])

# DENSE_LS: GMRES SIN PRECOND
for n in [500, 1000]:
    CASES.append(["--case","dense_ls","--n",str(n),"--method","GMRES","--restart","30","--precond","none","--tol","1e-10","--maxiter","4000"])

def run_case(args, append):
    cmd = [sys.executable, "-m", "src.experiments"] + args + ["--out", OUT]
    if append:
        cmd.append("--append")
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    for k, args in enumerate(CASES):
        run_case(args, append=(k>0))
    print("\nLISTO. CSV EN:", OUT)
