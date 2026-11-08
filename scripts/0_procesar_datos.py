# %% 1. Librerías y rutas
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# Se ejecuta desde la carpeta G4
SIN_PROCESAR = Path("./Datos sin procesar")
PROCESADOS = Path("./Datos Procesados")

# %% 2. Cargar Excel de Poblacion del DANE
def leer_tabla(ruta, hoja, columna_clave):
    crudo = pd.read_excel(ruta, sheet_name=hoja, header=None, dtype=str)
    limpio = crudo.apply(lambda col: col.str.replace(r"\s+", " ", regex=True).str.strip().str.upper())
    fila = limpio.index[(limpio == columna_clave).any(axis=1)][0]
    df = crudo.iloc[fila + 1:].copy()
    df.columns = limpio.iloc[fila]
    return df

# %% 3. Proyecciones 2018-2035.
p18 = leer_tabla(SIN_PROCESAR / "Poblacional 2018-2035.xlsx", 0, "ÁREA GEOGRÁFICA")
print("Filas originales:", len(p18))
print("Áreas geográficas:", p18["ÁREA GEOGRÁFICA"].value_counts(dropna=False).to_dict())

p18 = p18[p18["ÁREA GEOGRÁFICA"].str.strip().str.upper() == "TOTAL"]
p18 = pd.DataFrame({
    "divipola": p18["COD_DPTO-MPIO"],
    "municipio": p18["MUNICIPIO"],
    "anio": p18["AÑO"],
    "poblacion": p18["TOTAL"],
    "fuente": "Proyecciones 2018-2035",
})
print("Filas después de filtrar Total:", len(p18))
p18.head()

# %% 4. Retroproyección 2016
hojas = pd.ExcelFile(SIN_PROCESAR / "Poblacion 2016-18.xlsx").sheet_names
print("Hojas encontradas:", hojas)

partes = []
for hoja in hojas:
    t = leer_tabla(SIN_PROCESAR / "Poblacion 2016-18.xlsx", hoja, "POBLACIÓN TOTAL")
    print(f"  {hoja}: {len(t)} filas")
    partes.append(pd.DataFrame({
        "divipola": t["CÓDIGO MUNICIPIO"],
        "municipio": t["NOMBRE MUNICIPIO"],
        "anio": t["QUINCENA"].str.extract(r"(\d{4})")[0],  # sirve si viene como fecha o como texto
        "poblacion": t["POBLACIÓN TOTAL"],
        "fuente": "Retroproyección 31-dic",
    }))
p16 = pd.concat(partes)
p16.head()

# %% 5. Unificar y estandarizar Poblacion
# divipola: texto de 5 dígitos 
# anio y poblacion: numéricos.
pob = pd.concat([p18, p16], ignore_index=True)
pob["divipola"] = pob["divipola"].str.strip().str.zfill(5)
pob["anio"] = pd.to_numeric(pob["anio"], errors="coerce")
pob["poblacion"] = pd.to_numeric(pob["poblacion"], errors="coerce")
pob["municipio"] = pob["municipio"].str.strip()

filas_antes = len(pob)
pob = pob[pob["divipola"].str.fullmatch(r"\d{5}", na=False) & pob["anio"].notna()]
pob["anio"] = pob["anio"].astype(int)
print("Filas unidas:", filas_antes, "| descartadas (notas al pie / vacías):", filas_antes - len(pob))

# %% 6. Años repetidos entre archivos
# Si un municipio-año aparece en los dos archivos, se deja la proyección 2018-2035.
duplicados = pob.duplicated(["divipola", "anio"], keep=False)
print("Municipio-año repetidos entre archivos:", duplicados.sum() // 2)

pob["prioridad"] = (pob["fuente"] != "Proyecciones 2018-2035").astype(int)
pob = (pob.sort_values(["divipola", "anio", "prioridad"])
          .drop_duplicates(["divipola", "anio"])
          .drop(columns="prioridad")
          .reset_index(drop=True))

# %% 7. Resultado
print("Filas finales:", len(pob), "| municipios:", pob["divipola"].nunique())
print("\nVacíos por columna:\n", pob.isna().sum())
print("\nFilas por año y fuente:")
print(pob.groupby(["anio", "fuente"]).size().unstack(fill_value=0))

valle = ["05001", "05079", "05088", "05129", "05212", "05266", "05308", "05360", "05380", "05631"]
print("\nValle de Aburrá (población por año):")
print(pob[pob["divipola"].isin(valle)].pivot_table(index="municipio", columns="anio",
                                                    values="poblacion", aggfunc="sum"))

# %% 8. Guardar en Datos Procesados (solo si el archivo no existe)
PROCESADOS.mkdir(exist_ok=True)
salida = PROCESADOS / "DANE Poblacion 2016+2018-2035.xlsx"

if salida.exists():
    print("El archivo ya existe: ", salida)
else:
    pob.to_excel(salida, sheet_name="poblacion", index=False)
    print("Guardado en:", salida, "|", len(pob), "filas")

# %%
# DATOS ELIC


# %% 9. Leer ELIC
elic = pd.read_excel(SIN_PROCESAR / "ELIC.xlsx", sheet_name="elic",
                     dtype={"cod_depto": str, "cod_muni": str, "mes": str})
elic.columns = elic.columns.str.strip().str.lower()
print("Filas originales:", len(elic))
print("Columnas:", list(elic.columns))
elic.head()

# %% 10. Estandarizar
# - divipola = cod_depto (2 dígitos) + cod_muni (3 dígitos) -> "05" + "001" = "05001"
# - año y mes a enteros; área, unidades y licencias a numéricos.
elic = elic.rename(columns={"año": "anio"})
elic["divipola"] = elic["cod_depto"].str.strip().str.zfill(2) + elic["cod_muni"].str.strip().str.zfill(3)
elic["anio"] = pd.to_numeric(elic["anio"], errors="coerce").astype("Int64")
elic["mes"] = pd.to_numeric(elic["mes"], errors="coerce").astype("Int64")
for col in ["area", "unidades", "licencias"]:
    elic[col] = pd.to_numeric(elic[col], errors="coerce")

elic = elic[["divipola"] + [c for c in elic.columns if c != "divipola"]]

# %% 11. Resultado
print("Filas:", len(elic), "| municipios:", elic["divipola"].nunique(),
      "| años:", elic["anio"].min(), "-", elic["anio"].max())
print("Filas duplicadas exactas:", elic.duplicated().sum())
print("\nVacíos por columna:\n", elic.isna().sum())
print("\nCobertura por año (municipios en la muestra / valor de la columna 'cobertura'):")
print(elic.groupby("anio").agg(municipios=("divipola", "nunique"),
                               cobertura=("cobertura", "unique")))

print("\nValle de Aburrá - licencias por año:")
print(elic[elic["divipola"].isin(valle)].pivot_table(index="divipola", columns="anio",
                                                      values="licencias", aggfunc="sum"))

# %% 12. Guardar en "Datos Procesados" (solo si el archivo no existe)
salida = PROCESADOS / "DANE ELIC.xlsx"

if salida.exists():
    print("El archivo ya existe: ", salida)
else:
    elic.to_excel(salida, sheet_name="elic", index=False)
    print("Guardado en:", salida, "|", len(elic), "filas")
# %%
