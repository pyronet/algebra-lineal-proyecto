import os, glob
import pandas as pd

BASE = r"C:/Users/Administrator/IdeaProjects/algebra-lineal-proyecto"
PATTERN = os.path.join(BASE, "report", "*.csv")  # UNE TODOS LOS CSV DE /report
OUT_DIR = os.path.join(BASE, "report")
os.makedirs(OUT_DIR, exist_ok=True)

files = [f for f in glob.glob(PATTERN) if os.path.basename(f).lower().startswith("resultados_completos")]
if not files:
    raise SystemExit("NO SE ENCONTRARON CSV PARA UNIR EN /report")

dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["__origen__"] = os.path.basename(f)
        dfs.append(df)
    except Exception as e:
        print(f"AVISO: NO PUDE LEER {f}: {e}")

if not dfs:
    raise SystemExit("NO HAY DATOS VALIDOS")

full = pd.concat(dfs, ignore_index=True)

# ORDEN DE COLUMNAS SI EXISTEN
cols_order = ["Caso","n","Tipo","Metodo","Prec.","Precond.","Iter.","||r||2 final","Tiempo (s)","tol(rtol)","maxiter","restart","__origen__"]
full = full.reindex(columns=[c for c in cols_order if c in full.columns])

# PEQUENO FORMATEO
def fmt_scientific(x):
    try:
        return f"{float(x):.3e}"
    except Exception:
        return x

if "||r||2 final" in full.columns:
    full["||r||2 final"] = full["||r||2 final"].apply(fmt_scientific)
if "Tiempo (s)" in full.columns:
    full["Tiempo (s)"] = full["Tiempo (s)"].astype(float).map(lambda v: f"{v:.4f}")

# ORDEN SUGERIDO PARA LECTURA
sort_cols = [c for c in ["Tipo","Metodo","n","Iter."] if c in full.columns]
if sort_cols:
    full = full.sort_values(sort_cols)

print("\n=== TABLA UNIFICADA ===")
print(full.to_string(index=False))

# SALIDAS
out_csv  = os.path.join(OUT_DIR, "tabla_resultados_full.csv")
out_xlsx = os.path.join(OUT_DIR, "tabla_resultados_full.xlsx")
out_md   = os.path.join(OUT_DIR, "tabla_resultados_full.md")

full.to_csv(out_csv, index=False, encoding="utf-8")
full.to_excel(out_xlsx, index=False)
full.to_markdown(out_md, index=False)

print(f"\nARCHIVOS GENERADOS:")
print(f"- {out_csv}")
print(f"- {out_xlsx}")
print(f"- {out_md}")
