# %% 1. Librerías y rutas
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

PROCESADOS = Path("./Datos Procesados")
print([p.name for p in PROCESADOS.iterdir()])

# %% 2. IGAC (parquet reducido si existe; si no, el nacional)
igac_reducido = PROCESADOS / "IGAC 2015-2023 Valle de Aburra aledanos Bogota.parquet"

if igac_reducido.exists():
    igac = pd.read_parquet(igac_reducido)
    print("Leído el parquet reducido:", igac_reducido.name)
else:
    igac_nacional = next(p for p in PROCESADOS.glob("*.parquet") if p != igac_reducido)
    igac = pd.read_parquet(igac_nacional)
    print("Leído el parquet nacional:", igac_nacional.name)

print("IGAC:", igac.shape)
igac.info()
igac.head()
# %% 3. Población
pob = pd.read_excel(PROCESADOS / "DANE Poblacion 2016+2018-2035.xlsx", dtype={"divipola": str})
print("Población:", pob.shape)
pob.info()
pob.head()

# %% 4. ELIC
elic = pd.read_excel(PROCESADOS / "DANE ELIC.xlsx", dtype={"divipola": str, "cod_depto": str, "cod_muni": str})
print("ELIC:", elic.shape)
elic.info()
elic.head()
# %% 5. Características generales de cada set
# Objetivo: ver tamaño, años y municipios de cada fuente para decidir cómo reducir el IGAC.

# Municipios candidatos (DIVIPOLA)
candidatos = {
    # Valle de Aburrá
    "05001": "Medellín", "05079": "Barbosa", "05088": "Bello", "05129": "Caldas",
    "05212": "Copacabana", "05266": "Envigado", "05308": "Girardota",
    "05360": "Itagüí", "05380": "La Estrella", "05631": "Sabaneta",
    # Aledaños (Oriente cercano)
    "05615": "Rionegro", "05318": "Guarne", "05376": "La Ceja",
    "05440": "Marinilla", "05607": "El Retiro",
    # Referencia nacional
    "11001": "Bogotá",
}

# Columnas de año y municipio de cada set (el DIVIPOLA del IGAC se pasa a texto de 5 dígitos)
sets = {
    "IGAC": (igac["YEAR_RADICA"], igac["DIVIPOLA"].astype(str).str.zfill(5)),
    "Población": (pob["anio"], pob["divipola"]),
    "ELIC": (elic["anio"], elic["divipola"]),
}

# 1. Tamaño, años y municipios
general = pd.DataFrame({
    nombre: {
        "filas": len(anio),
        "año mínimo": anio.min(),
        "año máximo": anio.max(),
        "años distintos": anio.nunique(),
        "municipios distintos": div.nunique(),
    }
    for nombre, (anio, div) in sets.items()
})
print("1. Características generales")
print(general, "\n")

# 2. Filas por año en cada set
por_anio = pd.DataFrame({nombre: anio.value_counts() for nombre, (anio, div) in sets.items()}).sort_index()
print("2. Filas por año")
print(por_anio.fillna(0).astype(int), "\n")

# 3. Presencia de los municipios candidatos (filas en cada set)
presencia = pd.DataFrame({
    nombre: div[div.isin(candidatos)].value_counts()
    for nombre, (anio, div) in sets.items()
}).reindex(list(candidatos)).fillna(0).astype(int)
presencia.insert(0, "municipio", presencia.index.map(candidatos))
print("3. Municipios candidatos (filas por set)")
print(presencia)
print("\nFilas IGAC si se filtra a los candidatos:", presencia["IGAC"].sum(),
      f"({100 * presencia['IGAC'].sum() / len(igac):.1f} % del total)")

# %% 6. Reducir IGAC a los municipios candidatos y guardar parquet (solo si no existe)
# El DIVIPOLA del IGAC es entero (5001), así que se filtra con los códigos como número.
codigos = [int(c) for c in candidatos]

salida_igac = PROCESADOS / "IGAC 2015-2023 Valle de Aburra aledanos Bogota.parquet"

if salida_igac.exists():
    print("El archivo ya existe, no se sobrescribe:", salida_igac)
else:
    igac_red = igac[igac["DIVIPOLA"].isin(codigos)].reset_index(drop=True)
    igac_red.to_parquet(salida_igac, index=False)
    print("Guardado en:", salida_igac)
    print("Filas:", len(igac_red), f"({100 * len(igac_red) / len(igac):.1f} % del original)",
          "| municipios:", igac_red["DIVIPOLA"].nunique())
    print(f"Memoria: {igac_red.memory_usage(deep=True).sum() / 1e9:.2f} GB "
          f"(antes {igac.memory_usage(deep=True).sum() / 1e9:.1f} GB)")

# %% 7. Validación de calidad por set
# 4 características totalidad, consistencia, claridad, formato

def perfil(df):
    """Formato y totalidad por columna: tipo, % de vacíos, valores distintos y un ejemplo."""
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "% vacíos": (100 * df.isna().mean()).round(2),
        "distintos": df.nunique(),
        "ejemplo": df.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else None),
    })

# ========================= IGAC =========================
print("=" * 25, "IGAC", "=" * 25)
print(perfil(igac), "\n")

# Totalidad: registros por municipio y año (un 0 = municipio-año sin datos)
div_igac = igac["DIVIPOLA"].astype(str).str.zfill(5)
conteo = pd.crosstab(div_igac.map(candidatos), igac["YEAR_RADICA"])
print("Registros por municipio y año:\n", conteo)
print("Municipio-año sin registros:", int((conteo == 0).sum().sum()))
print("% de registros con valor por año:\n",
      (100 * igac.groupby("YEAR_RADICA")["TIENE_VALOR"].mean()).round(1).to_dict(), "\n")

# Consistencia
valor_num = pd.to_numeric(igac["VALOR"], errors="coerce")
fecha = pd.to_datetime(igac["FECHA_RADICA_TEXTO"], errors="coerce")
print("TIENE_VALOR = 1 pero VALOR vacío:", int(((igac["TIENE_VALOR"] == 1) & valor_num.isna()).sum()))
print("TIENE_VALOR = 0 pero VALOR con dato:", int(((igac["TIENE_VALOR"] == 0) & valor_num.notna()).sum()))
print("VALOR con texto no numérico:", int((igac["VALOR"].notna() & valor_num.isna()).sum()))
print("VALOR <= 0:", int((valor_num <= 0).sum()))
print("Registros con más de un valor:", int(igac["TIENE_MAS_DE_UN_VALOR"].sum()))
print("PK duplicadas:", int(igac["PK"].duplicated().sum()))
print("Fechas de radicación no convertibles:", int(fecha.isna().sum()))
print("Año de la fecha distinto de YEAR_RADICA:",
      int((fecha.notna() & (fecha.dt.year != igac["YEAR_RADICA"])).sum()))
print("Máx. nombres de municipio por DIVIPOLA:", igac.groupby("DIVIPOLA")["MUNICIPIO"].nunique().max())
print("Valores de Dinámica_Inmobiliaria:", igac["Dinámica_Inmobiliaria"].value_counts().to_dict(), "\n")

# ======================= POBLACIÓN =======================
print("=" * 25, "POBLACIÓN", "=" * 25)
print(perfil(pob), "\n")

# Totalidad: años y municipios por año
print("Años faltantes entre 2015 y 2035:", sorted(set(range(2015, 2036)) - set(pob["anio"])))
print("Municipios por año:", pob.groupby("anio")["divipola"].nunique().to_dict())
m16 = set(pob.loc[pob["anio"] == 2016, "divipola"])
m18 = set(pob.loc[pob["anio"] == 2018, "divipola"])
m19 = set(pob.loc[pob["anio"] == 2019, "divipola"])
print("Municipios de 2019 que no están en 2016:", len(m19 - m16))
print("Municipios de 2018 que no están en 2019:", sorted(m18 - m19), "\n")

# Consistencia
print("Duplicados municipio-año:", int(pob.duplicated(["divipola", "anio"]).sum()))
print("Población <= 0:", int((pob["poblacion"] <= 0).sum()))
print("DIVIPOLA con longitud distinta de 5:", int((pob["divipola"].str.len() != 5).sum()))
pc = pob[pob["divipola"].isin(candidatos)].pivot_table(index="divipola", columns="anio", values="poblacion")
salto = (100 * pc.pct_change(axis=1)).round(1).rename(index=candidatos)
print("Variación % de la población en candidatos (2018 = cambio frente a 2016, cambio de fuente):\n",
      salto[[a for a in [2018, 2019, 2020] if a in salto.columns]], "\n")

# ========================= ELIC =========================
print("=" * 25, "ELIC", "=" * 25)
print(perfil(elic), "\n")

# Totalidad: meses con registro por municipio-año (ELIC solo trae filas cuando hubo licencias)
meses = (elic[elic["divipola"].isin(candidatos) & elic["anio"].between(2015, 2023)]
         .groupby(["divipola", "anio"])["mes"].nunique().unstack(fill_value=0).rename(index=candidatos))
print("Meses con registro por municipio y año (2015-2023):\n", meses, "\n")

# Consistencia y claridad
print("Filas duplicadas exactas:", int(elic.duplicated().sum()))
print("Mes fuera de 1-12:", int((~elic["mes"].between(1, 12)).sum()))
print("Valores negativos:", (elic[["area", "unidades", "licencias"]] < 0).sum().to_dict())
print("Filas con 0 licencias:", int((elic["licencias"] == 0).sum()))
print("Valores de cobertura:", elic["cobertura"].value_counts().to_dict())
print("Códigos por variable categórica (se interpretan con la hoja guía):")
for col in ["obj_tra", "clase_suelo", "modalidad", "destino", "tipo_vivi", "vis_nvis", "estrato"]:
    print(f"  {col}: {sorted(elic[col].unique().tolist())}")

# %% 8. Diagnóstico de los hallazgos de la validación


def patron(s):
    """Reemplaza cada dígito por 9 para ver la forma del texto (ej. '1.234.567' -> '9.999.999')."""
    return s.astype(str).str.replace(r"\d", "9", regex=True)

# ---------- 1. IGAC: VALOR que no se puede convertir a número ----------
print("=" * 20, "1. IGAC - VALOR no numérico", "=" * 20)
valor_texto = igac.loc[igac["VALOR"].notna() & valor_num.isna(), "VALOR"]
print("Registros:", len(valor_texto))
print("\nPatrones más frecuentes:\n", patron(valor_texto).value_counts().head(15))
print("\nEjemplos:\n", valor_texto.drop_duplicates().head(10).tolist(), "\n")

# ---------- 2. IGAC: fechas de radicación que no se pueden convertir ----------
print("=" * 20, "2. IGAC - FECHA_RADICA no convertible", "=" * 20)
fecha_texto = igac.loc[fecha.isna(), "FECHA_RADICA_TEXTO"]
print("Registros:", len(fecha_texto))
print("\nPatrones más frecuentes:\n", patron(fecha_texto).value_counts().head(10))
print("\nPatrones de las fechas que sí se convirtieron:\n",
      patron(igac.loc[fecha.notna(), "FECHA_RADICA_TEXTO"].head(100000)).value_counts().head(5))
print("\nEjemplos:\n", fecha_texto.drop_duplicates().head(10).tolist(), "\n")

# ---------- 3. Población: duplicados, población <= 0 y años por fuente ----------
print("=" * 20, "3. POBLACIÓN", "=" * 20)
print("Filas por año y fuente:\n", pd.crosstab(pob["anio"], pob["fuente"]).head(5))
dup = pob[pob.duplicated(["divipola", "anio"], keep=False)].sort_values(["divipola", "anio"])
print("\nDuplicados municipio-año (", len(dup), "filas):\n", dup.head(12))
print("\nPoblación <= 0:\n", pob[pob["poblacion"] <= 0])
grandes = ["05001", "05088", "05266", "05360", "11001"]
print("\nMunicipios grandes (deberían tener 2016):\n",
      pob[pob["divipola"].isin(grandes) & (pob["anio"] <= 2018)].sort_values(["divipola", "anio"]), "\n")

# ---------- 4. ELIC: filas con 0 licencias, cobertura y códigos VIS ----------
print("=" * 20, "4. ELIC", "=" * 20)
cero = elic[elic["licencias"] == 0]
print("Filas con 0 licencias:", len(cero),
      "| con área > 0:", int((cero["area"] > 0).sum()),
      "| con unidades > 0:", int((cero["unidades"] > 0).sum()))
print("\nCobertura por año:\n", pd.crosstab(elic["anio"], elic["cobertura"]))
print("\nCódigos de vis_nvis según destino (filas):\n", pd.crosstab(elic["destino"], elic["vis_nvis"]))

# %%
