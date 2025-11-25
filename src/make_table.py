import os
import pandas as pd

# RUTA DE ENTRADA Y SALIDAS (AJUSTA SI QUIERES)
IN_CSV  = r"C:/Users/Administrator/IdeaProjects/algebra-lineal-proyecto/report/tabla_resultados.csv"
OUT_DIR = r"C:/Users/Administrator/IdeaProjects/algebra-lineal-proyecto/report"

os.makedirs(OUT_DIR, exist_ok=True)

# LEE CSV
df = pd.read_csv(IN_CSV)

# OPCIONAL: ORDEN Y TIPOS
cols_order = ["Caso","n","Tipo","Metodo","Prec.","Precond.","Iter.","||r||2 final","Tiempo (s)","tol(rtol)","maxiter","restart"]
df = df.reindex(columns=[c for c in cols_order if c in df.columns])

# FORMATEOS SUAVES PARA VISUALIZAR
def fmt_scientific(x):
    try:
        return f"{float(x):.3e}"
    except Exception:
        return x

if "||r||2 final" in df.columns:
    df["||r||2 final"] = df["||r||2 final"].apply(fmt_scientific)
if "Tiempo (s)" in df.columns:
    df["Tiempo (s)"] = df["Tiempo (s)"].astype(float).map(lambda v: f"{v:.4f}")

# MUESTRA EN CONSOLA
print("\n=== TABLA DE RESULTADOS ===")
print(df.to_string(index=False))

# GUARDA VERSIONES
out_csv = os.path.join(OUT_DIR, "tabla_resultados_limpio.csv")
out_xlsx = os.path.join(OUT_DIR, "tabla_resultados.xlsx")
out_md = os.path.join(OUT_DIR, "tabla_resultados.md")

df.to_csv(out_csv, index=False, encoding="utf-8")
df.to_excel(out_xlsx, index=False)
df.to_markdown(out_md, index=False)

print(f"\nARCHIVOS GENERADOS:")
print(f"- {out_csv}")
print(f"- {out_xlsx}")
print(f"- {out_md}")
