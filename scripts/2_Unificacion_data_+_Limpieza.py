# Limpieza y construcción de los sets de análisis
# Dos sets para 2021-2023 y 16 municipios
# - Transacciones: IGAC a nivel de registro, limpio.
# - Summary: una fila por municipio-año con IGAC + Población + ELIC.

# %% 1. Librerías, rutas y parámetros
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

PROCESADOS = Path("./Datos Procesados")
UNIFICADOS = Path("./Datos unificados")

ANIOS = [2021, 2022, 2023]
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

# %% 2. Leer los tres sets (IGAC solo 2021-2023)
igac = pd.read_parquet(PROCESADOS / "IGAC 2015-2023 Valle de Aburra aledanos Bogota.parquet",
                       filters=[("YEAR_RADICA", "in", ANIOS)])
pob = pd.read_excel(PROCESADOS / "DANE Poblacion 2016+2018-2035.xlsx", dtype={"divipola": str})
elic = pd.read_excel(PROCESADOS / "DANE ELIC.xlsx", dtype={"divipola": str})
print("IGAC 2021-2023:", igac.shape, "| Población:", pob.shape, "| ELIC:", elic.shape)

# %% 3. IGAC - corregir formatos (hallazgos del punto 1)
# - DIVIPOLA: entero -> texto de 5 dígitos (5001 -> "05001").
# - VALOR: texto con comas de miles ("530,000,000") -> número. VALOR_VALIDO marca los mayores que 0.
# - COUNT_A, COUNT_DE, NUM_ANOTACION: texto -> número.
# - NOMBRE_NATUJUR: mayúsculas y sin espacios sobrantes.
# - Dinámica_Inmobiliaria -> DINAMICA_INMOBILIARIA (mismo estilo que las demás columnas).
igac["DIVIPOLA"] = igac["DIVIPOLA"].astype(str).str.zfill(5)
igac["VALOR"] = pd.to_numeric(igac["VALOR"].str.replace(",", "", regex=False), errors="coerce")
igac["VALOR_VALIDO"] = igac["VALOR"] > 0
for col in ["COUNT_A", "COUNT_DE", "NUM_ANOTACION"]:
    igac[col] = pd.to_numeric(igac[col], errors="coerce").astype("Int64")
igac["NOMBRE_NATUJUR"] = igac["NOMBRE_NATUJUR"].str.upper().str.strip()
igac = igac.rename(columns={"Dinámica_Inmobiliaria": "DINAMICA_INMOBILIARIA"})

print("VALOR con dato:", int(igac["VALOR"].notna().sum()),
      "| mayor que 0:", int(igac["VALOR_VALIDO"].sum()),
      "| igual o menor que 0:", int((igac["VALOR"] <= 0).sum()))

# %% 4. IGAC - unificar las fechas (3 formatos en la misma columna)
# "2018-02-05 00:00:00", "30/06/2015" y "16/11/22" -> una sola fecha.
def convertir_fecha(texto):
    fecha = pd.to_datetime(texto, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    fecha = fecha.fillna(pd.to_datetime(texto, format="%d/%m/%Y", errors="coerce"))
    fecha = fecha.fillna(pd.to_datetime(texto, format="%d/%m/%y", errors="coerce"))
    return fecha

igac["FECHA_RADICA"] = convertir_fecha(igac["FECHA_RADICA_TEXTO"])
igac["FECHA_APERTURA"] = convertir_fecha(igac["FECHA_APERTURA_TEXTO"])

print("FECHA_RADICA sin convertir:", int(igac["FECHA_RADICA"].isna().sum()))
print("FECHA_APERTURA sin convertir (con texto):",
      int((igac["FECHA_APERTURA_TEXTO"].notna() & igac["FECHA_APERTURA"].isna()).sum()))
print("Año de FECHA_RADICA distinto de YEAR_RADICA:",
      int((igac["FECHA_RADICA"].dt.year != igac["YEAR_RADICA"]).sum()))

# %% 5. IGAC - quitar columnas vacías y guardar el set de transacciones
# - Se eliminan las columnas 100 % vacías (en el punto 1: NUMERO_CATASTRAL y NUMERO_CATASTRAL_ANTIGUO).
# - Se eliminan las columnas de fecha en texto (ya quedaron convertidas).
# - FOLIOS_EN_DOCUMENTO: cuántos folios tiene la misma escritura (actos con varios inmuebles).
vacias = [c for c in igac.columns if igac[c].isna().all()]
igac = igac.drop(columns=vacias + ["FECHA_RADICA_TEXTO", "FECHA_APERTURA_TEXTO"])
igac["FOLIOS_EN_DOCUMENTO"] = (igac.groupby(["DIVIPOLA", "DOCUMENTO_JUSTIFICATIVO"])["MATRICULA"]
                                   .transform("nunique"))
igac = igac.reset_index(drop=True)

print("Columnas eliminadas por estar vacías:", vacias)
print("Filas duplicadas exactas:", int(igac.duplicated().sum()))
print("IGAC limpio:", igac.shape)
igac.info()
# %%
salida_igac = UNIFICADOS / "IGAC transacciones 2021-2023 limpio.parquet"
UNIFICADOS.mkdir(exist_ok=True)

if salida_igac.exists():
    print("\nEl archivo ya existe, no se sobrescribe:", salida_igac)
else:
    igac.to_parquet(salida_igac, index=False)
    print("\nGuardado en:", salida_igac, "|", len(igac), "filas")

# %% 6. Set summary (una fila por municipio-año)
# IGAC (todos los actos): registros, dinámica y registros con valor.
# Población: proyección DANE. ELIC: solo vivienda (destino = 1).
# %% 6. Set summary (una fila por municipio-año)
# IGAC (todos los actos): registros, dinámica y registros con valor.
# Población: proyección DANE. ELIC: solo vivienda (destino = 1).

# IGAC
s_igac = igac.groupby(["DIVIPOLA", "YEAR_RADICA"]).agg(
    registros=("PK", "size"),
    dinamica=("DINAMICA_INMOBILIARIA", "sum"),
    registros_con_valor=("VALOR_VALIDO", "sum"),
)
s_igac.index.names = ["divipola", "anio"]

# Población
s_pob = (pob[pob["anio"].isin(ANIOS) & pob["divipola"].isin(candidatos)]
         .drop_duplicates(["divipola", "anio"])
         .set_index(["divipola", "anio"])[["poblacion"]])

# ELIC: códigos según la hoja guía (0 = no aplica)
CODIGOS_VIS = {1: "vis", 2: "novis", 3: "vip"}       # vis_nvis: VIS (sin VIP), No VIS, VIP
CODIGOS_TIPO = {1: "casas", 2: "apartamentos"}        # tipo_vivi

viv = elic[elic["anio"].isin(ANIOS) & elic["divipola"].isin(candidatos) & (elic["destino"] == 1)].copy()
# Obra nueva = trámite nuevo (obj_tra = 1) y construcción nueva (modalidad = 1); excluye modificaciones y ampliaciones
viv["unidades_obra_nueva"] = viv["unidades"].where((viv["obj_tra"] == 1) & (viv["modalidad"] == 1), 0)

s_elic = viv.groupby(["divipola", "anio"]).agg(
    licencias_vivienda=("licencias", "sum"),
    unidades_vivienda=("unidades", "sum"),
    unidades_obra_nueva=("unidades_obra_nueva", "sum"),
    area_vivienda_m2=("area", "sum"),
)

def unidades_por(columna, codigos):
    """Unidades de vivienda separadas por las categorías de una columna codificada."""
    return (viv.assign(tipo=viv[columna].map(codigos))
               .dropna(subset=["tipo"])
               .pivot_table(index=["divipola", "anio"], columns="tipo", values="unidades",
                            aggfunc="sum", fill_value=0)
               .add_prefix("unidades_"))

s_vis = unidades_por("vis_nvis", CODIGOS_VIS)
s_tipo = unidades_por("tipo_vivi", CODIGOS_TIPO)

# Unir y calcular indicadores
summary = pd.concat([s_pob, s_igac, s_elic, s_vis, s_tipo], axis=1).reset_index()
summary.insert(1, "municipio", summary["divipola"].map(candidatos))
summary["pct_dinamica"] = (100 * summary["dinamica"] / summary["registros"]).round(1)
summary["pct_registros_con_valor"] = (100 * summary["registros_con_valor"] / summary["registros"]).round(1)
for col in ["registros", "dinamica", "licencias_vivienda", "unidades_vivienda", "unidades_obra_nueva"]:
    summary[f"{col}_x1000hab"] = (1000 * summary[col] / summary["poblacion"]).round(2)

print("Filas:", len(summary), "(esperadas:", len(candidatos) * len(ANIOS), ")")
print("Vacíos por columna:\n", summary.isna().sum()[summary.isna().sum() > 0])
summary

# %% 7. Guardar summary en "Datos unificados" (solo si no existe)
UNIFICADOS.mkdir(exist_ok=True)
salida_summary = UNIFICADOS / "Summary municipio-anio 2021-2023.xlsx"

if salida_summary.exists():
    print("El archivo ya existe, no se sobrescribe:", salida_summary)
else:
    summary.to_excel(salida_summary, index=False)
    print("Guardado en:", salida_summary, "|", len(summary), "filas")
# %%
# %% 8. Verificación de los sets limpios


trans = pd.read_parquet(UNIFICADOS / "IGAC transacciones 2021-2023 limpio.parquet")
summ = pd.read_excel(UNIFICADOS / "Summary municipio-anio 2021-2023.xlsx", dtype={"divipola": str})

revisiones = []
def revisar(set_datos, regla, valor, ok):
    revisiones.append({"set": set_datos, "revisión": regla, "resultado": valor, "estado": "OK" if ok else "REVISAR"})

# ---------- Transacciones ----------
T = "Transacciones"
revisar(T, "Filas", len(trans), len(trans) > 0)
revisar(T, "Años distintos de 2021-2023", int((~trans["YEAR_RADICA"].isin(ANIOS)).sum()),
        trans["YEAR_RADICA"].isin(ANIOS).all())
revisar(T, "Municipios fuera de los 16 candidatos", int((~trans["DIVIPOLA"].isin(candidatos)).sum()),
        trans["DIVIPOLA"].isin(candidatos).all())
revisar(T, "Municipios presentes", trans["DIVIPOLA"].nunique(), trans["DIVIPOLA"].nunique() == len(candidatos))
revisar(T, "DIVIPOLA con longitud distinta de 5", int((trans["DIVIPOLA"].str.len() != 5).sum()),
        (trans["DIVIPOLA"].str.len() == 5).all())
revisar(T, "PK duplicadas", int(trans["PK"].duplicated().sum()), not trans["PK"].duplicated().any())
revisar(T, "Filas duplicadas exactas", int(trans.duplicated().sum()), not trans.duplicated().any())
revisar(T, "Columnas 100 % vacías", [c for c in trans.columns if trans[c].isna().all()],
        not trans.isna().all().any())
revisar(T, "VALOR es numérico", str(trans["VALOR"].dtype), pd.api.types.is_numeric_dtype(trans["VALOR"]))
revisar(T, "TIENE_VALOR = 1 sin VALOR", int(((trans["TIENE_VALOR"] == 1) & trans["VALOR"].isna()).sum()),
        not ((trans["TIENE_VALOR"] == 1) & trans["VALOR"].isna()).any())
revisar(T, "VALOR_VALIDO coincide con VALOR > 0", int((trans["VALOR_VALIDO"] != (trans["VALOR"] > 0)).sum()),
        (trans["VALOR_VALIDO"] == (trans["VALOR"] > 0)).all())
revisar(T, "VALOR <= 0 (marcados, no eliminados)", int((trans["VALOR"] <= 0).sum()), True)
revisar(T, "FECHA_RADICA es fecha", str(trans["FECHA_RADICA"].dtype),
        pd.api.types.is_datetime64_any_dtype(trans["FECHA_RADICA"]))
revisar(T, "FECHA_RADICA vacía", int(trans["FECHA_RADICA"].isna().sum()), trans["FECHA_RADICA"].notna().all())
revisar(T, "Año de FECHA_RADICA distinto de YEAR_RADICA",
        int((trans["FECHA_RADICA"].dt.year != trans["YEAR_RADICA"]).sum()),
        (trans["FECHA_RADICA"].dt.year == trans["YEAR_RADICA"]).all())
revisar(T, "DINAMICA_INMOBILIARIA solo 0 y 1", sorted(trans["DINAMICA_INMOBILIARIA"].unique().tolist()),
        trans["DINAMICA_INMOBILIARIA"].isin([0, 1]).all())

# ---------- Summary ----------
S = "Summary"
revisar(S, "Filas", len(summ), len(summ) == len(candidatos) * len(ANIOS))
revisar(S, "Municipio-año duplicados", int(summ.duplicated(["divipola", "anio"]).sum()),
        not summ.duplicated(["divipola", "anio"]).any())
revisar(S, "Celdas vacías", int(summ.isna().sum().sum()), not summ.isna().any().any())
numericas = summ.select_dtypes("number")
revisar(S, "Valores negativos", int((numericas < 0).sum().sum()), not (numericas < 0).any().any())
revisar(S, "dinamica > registros", int((summ["dinamica"] > summ["registros"]).sum()),
        (summ["dinamica"] <= summ["registros"]).all())
revisar(S, "registros_con_valor > registros", int((summ["registros_con_valor"] > summ["registros"]).sum()),
        (summ["registros_con_valor"] <= summ["registros"]).all())
suma_vis = summ[["unidades_vis", "unidades_novis", "unidades_vip"]].sum(axis=1)
revisar(S, "VIS + No VIS + VIP distinto de unidades_vivienda", int((suma_vis != summ["unidades_vivienda"]).sum()),
        (suma_vis == summ["unidades_vivienda"]).all())
suma_tipo = summ[["unidades_casas", "unidades_apartamentos"]].sum(axis=1)
revisar(S, "Casas + apartamentos distinto de unidades_vivienda", int((suma_tipo != summ["unidades_vivienda"]).sum()),
        (suma_tipo == summ["unidades_vivienda"]).all())
revisar(S, "unidades_obra_nueva > unidades_vivienda",
        int((summ["unidades_obra_nueva"] > summ["unidades_vivienda"]).sum()),
        (summ["unidades_obra_nueva"] <= summ["unidades_vivienda"]).all())

# ---------- Cruce entre los dos sets ----------
desde_trans = trans.groupby(["DIVIPOLA", "YEAR_RADICA"]).size()
desde_summ = summ.set_index(["divipola", "anio"])["registros"]
desde_trans.index.names = desde_summ.index.names
diferencias = int((desde_trans.reindex(desde_summ.index) != desde_summ).sum())
revisar("Cruce", "Registros del summary distintos a los de transacciones", diferencias, diferencias == 0)

# ---------- Resultado ----------
tabla = pd.DataFrame(revisiones)
print(tabla.to_string(index=False))
print("\nRevisiones OK:", (tabla["estado"] == "OK").sum(), "de", len(tabla))

# Valores extremos del VALOR (evidencia de la causa c; el tratamiento con IQR va en el punto 3)
validos = trans.loc[trans["VALOR_VALIDO"], "VALOR"]
print("\nPercentiles del VALOR (solo valores > 0):")
print(validos.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]).map("{:,.0f}".format))
# %%
