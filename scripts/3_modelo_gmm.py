"""Construye el modelo GMM del anteproyecto y exporta resultados verificables.

Metodología alineada con la Guía 4: siete indicadores, StandardScaler,
PCA con dos componentes y GMM con tres grupos (por defecto).
No es un modelo de recomendación financiera ni un ranking de municipios.
"""

from pathlib import Path
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# CONFIGURACIÓN DE VARIABLES Y RUTAS
#-------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
RUTAS_ENTRADA = [
    RAIZ / "datos" / "Summary municipio-anio 2021-2023.xlsx",
    RAIZ / "dashboard" / "data" / "Summary municipio-anio 2021-2023.xlsx"]
DIRECTORIO_SALIDA = RAIZ / "dashboard" / "data"

INDICADORES = [
    "registros_x1000hab",
    "dinamica_x1000hab",
    "pct_dinamica",
    "pct_registros_con_valor",
    "licencias_vivienda_x1000hab",
    "unidades_vivienda_x1000hab",
    "unidades_obra_nueva_x1000hab"]

NOMBRES = {
    "registros_x1000hab": "Actos registrales por 1.000 habitantes",
    "dinamica_x1000hab": "Actos con dinámica por 1.000 habitantes",
    "pct_dinamica": "Porcentaje de registros con dinámica",
    "pct_registros_con_valor": "Porcentaje de registros con valor válido",
    "licencias_vivienda_x1000hab": "Licencias por 1.000 habitantes",
    "unidades_vivienda_x1000hab": "Unidades autorizadas por 1.000 habitantes",
    "unidades_obra_nueva_x1000hab": "Obra nueva autorizada por 1.000 habitantes"}

VALLE = {"05001", "05079", "05088", "05129", "05212", "05266", "05308", "05360", "05380", "05631"}


# CARGA Y VALIDACIÓN DE LOS DATOS
#-------------------------------------------------------------------
def cargar_datos(ruta):
    base = pd.read_excel(ruta, dtype={"divipola": str})
    requeridas = {"divipola", "municipio", "anio", *INDICADORES}
    faltantes = sorted(requeridas - set(base.columns))
    if faltantes:
        raise ValueError(f"Faltan variables del informe: {', '.join(faltantes)}")
    if base.empty:
        raise ValueError("La base integrada está vacía.")
    base["divipola"] = base["divipola"].astype(str).str.zfill(5)
    base["anio"] = pd.to_numeric(base["anio"], errors="raise").astype(int)
    if base.duplicated(["divipola", "anio"]).any():
        raise ValueError("Se encontraron combinaciones municipio-año repetidas.")
    if base[INDICADORES].isna().any().any():
        raise ValueError("Hay indicadores faltantes. Revisar antes de ajustar el modelo.")
    if not np.isfinite(base[INDICADORES].to_numpy(dtype=float)).all():
        raise ValueError("Los indicadores incluyen valores no finitos.")
    base["ambito"] = np.where(base["divipola"].isin(VALLE), "Valle de Aburrá", "Referencia")
    return base.sort_values(["divipola", "anio"]).reset_index(drop=True)


def obtener_ruta(ruta_usuario):
    if ruta_usuario:
        ruta = Path(ruta_usuario).expanduser().resolve()
        if not ruta.is_file():
            raise FileNotFoundError(f"No se encontró el archivo indicado: {ruta}")
        return ruta
    ruta = next((p for p in RUTAS_ENTRADA if p.is_file()), None)
    if ruta is None:
        raise FileNotFoundError("No está el resumen en datos/ ni en dashboard/data/.")
    return ruta


# EVALUACIÓN Y AJUSTE DE GMM SOBRE LOS DOS COMPONENTES PCA
#-------------------------------------------------------------------
def construir_modelo(base, numero_grupos=3, semilla=42):
    datos = base[INDICADORES].to_numpy(dtype=float)
    escalador = StandardScaler()
    estandarizados = escalador.fit_transform(datos)
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(estandarizados)

    resultados_evaluacion = []
    ajustes = {}
    for k in (2, 3, 4):
        if k >= len(base):
            continue
        mezcla = GaussianMixture(n_components=k, covariance_type="full",
            random_state=semilla, n_init=10, max_iter=500, reg_covar=0.001)
        mezcla.fit(componentes)
        grupos = mezcla.predict(componentes)
        silueta = float(silhouette_score(componentes, grupos)) if len(set(grupos)) > 1 else np.nan
        resultados_evaluacion.append({
            "metodo": "GMM", "grupos": k, "bic": float(mezcla.bic(componentes)),
            "aic": float(mezcla.aic(componentes)), "silueta": silueta,
            "grupos_efectivos": int(len(set(grupos))),
            "convergencia": bool(mezcla.converged_)})
        ajustes[k] = mezcla

    if numero_grupos not in ajustes:
        raise ValueError(f"No fue posible ajustar GMM con {numero_grupos} grupos.")
    modelo = ajustes[numero_grupos]
    etiquetas = modelo.predict(componentes)
    probabilidades = modelo.predict_proba(componentes)
    if not np.allclose(probabilidades.sum(axis=1), 1, atol=1e-9):
        raise ValueError("Las probabilidades GMM no suman 1 para todas las observaciones.")
    if not np.array_equal(etiquetas, np.argmax(probabilidades, axis=1)):
        raise ValueError("La clasificación GMM no coincide con las probabilidades calculadas.")

    # Comparamos K-Means y GMM en la misma proyección PCA para conservar la base de la Guía 4.
    for k in (2, 3, 4):
        if k >= len(base):
            continue
        agrupador = KMeans(n_clusters=k, random_state=semilla, n_init=10)
        grupos = agrupador.fit_predict(componentes)
        resultados_evaluacion.append({
            "metodo": "K-Means", "grupos": k, "bic": np.nan, "aic": np.nan,
            "silueta": float(silhouette_score(componentes, grupos)),
            "grupos_efectivos": int(len(set(grupos))), "convergencia": True})
    tabla_evaluacion = pd.DataFrame(resultados_evaluacion)

    resultados = base[["divipola", "municipio", "anio", "ambito"]].copy()
    resultados["cluster_gmm"] = etiquetas
    for j in range(numero_grupos):
        resultados[f"prob_cluster_{j}"] = probabilidades[:, j]
    resultados["probabilidad_maxima"] = probabilidades.max(axis=1)
    resultados["PC1"] = componentes[:, 0]
    resultados["PC2"] = componentes[:, 1]

    perfiles = base[["divipola", "anio", *INDICADORES]].copy()
    perfiles["cluster_gmm"] = etiquetas
    tabla_perfiles = perfiles.groupby("cluster_gmm", as_index=False).agg({
        **{col: "mean" for col in INDICADORES}, "divipola": "size"}).rename(columns={"divipola": "observaciones"})
    resumen_anual = resultados.groupby(["anio", "cluster_gmm"], as_index=False).agg(
        observaciones=("divipola", "size"), municipios=("divipola", "nunique"))

    # Prueba de estabilidad: comparamos asignaciones entre varias semillas.
    pruebas_estabilidad = []
    for semilla_alternativa in (7, 11, 21, 99):
        alternativa = GaussianMixture(n_components=numero_grupos, covariance_type="full",
            random_state=semilla_alternativa, n_init=10, max_iter=500, reg_covar=0.001)
        etiquetas_alt = alternativa.fit_predict(componentes)
        pruebas_estabilidad.append({"semilla": semilla_alternativa,
            "ari_frente_a_semilla_42": float(adjusted_rand_score(etiquetas, etiquetas_alt))})
    tabla_estabilidad = pd.DataFrame(pruebas_estabilidad)

    k_evaluado = tabla_evaluacion[(tabla_evaluacion["metodo"] == "GMM") &
        (tabla_evaluacion["grupos"] == numero_grupos)].iloc[0]
    metadatos = {
        "descripcion": "Segmentación exploratoria municipio-año; no ranking ni viabilidad financiera",
        "fuente": "Resumen integrado IGAC / DANE, 2021–2023",
        "observaciones": int(len(base)),
        "municipios": int(base["divipola"].nunique()),
        "anios": sorted(int(a) for a in base["anio"].unique()),
        "variables": INDICADORES,
        "descripciones_variables": NOMBRES,
        "preparacion": "StandardScaler sobre siete indicadores; PCA con dos componentes",
        "modelo": "GaussianMixture (EM)",
        "grupos": int(numero_grupos),
        "covarianza": "full",
        "semilla": int(semilla),
        "inicializaciones": 10,
        "reg_covar": 0.001,
        "convergencia": bool(modelo.converged_),
        "iteraciones_em": int(modelo.n_iter_),
        "varianza_pc1_pct": float(pca.explained_variance_ratio_[0] * 100),
        "varianza_pc2_pct": float(pca.explained_variance_ratio_[1] * 100),
        "varianza_pca2_pct": float(sum(pca.explained_variance_ratio_) * 100),
        "silueta": float(k_evaluado["silueta"]),
        "bic": float(k_evaluado["bic"]),
        "aic": float(k_evaluado["aic"]),
        "ari_minimo_semillas": float(tabla_estabilidad["ari_frente_a_semilla_42"].min()),
        "interpretacion": "Las probabilidades representan pertenencia estimada a grupos; no probabilidad de éxito inmobiliario"}
    objetos = {"escalador": escalador, "pca": pca, "gmm": modelo, "variables": INDICADORES}
    return resultados, tabla_evaluacion, tabla_perfiles, resumen_anual, tabla_estabilidad, metadatos, objetos


# EXPORTACIÓN DE RESULTADOS Y EVIDENCIAS
#-------------------------------------------------------------------
def exportar(resultados, evaluacion, perfiles, resumen_anual, estabilidad, metadatos, objetos, salida):
    salida.mkdir(parents=True, exist_ok=True)
    archivos = {
        "gmm_resultados.csv": resultados,
        "gmm_evaluacion.csv": evaluacion,
        "gmm_perfiles.csv": perfiles,
        "gmm_distribucion_anual.csv": resumen_anual,
        "gmm_estabilidad.csv": estabilidad}
    for nombre, tabla in archivos.items():
        tabla.to_csv(salida / nombre, index=False, encoding="utf-8-sig", float_format="%.10f")
    with (salida / "gmm_config.json").open("w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, ensure_ascii=False, indent=2)
    joblib.dump(objetos, salida / "gmm_modelo.joblib")
    return archivos


def main():
    parser = argparse.ArgumentParser(description="Reproduce GMM de la Guía 4 para visualizarlo en Streamlit.")
    parser.add_argument("--entrada", type=str, help="Ruta opcional del resumen Excel")
    parser.add_argument("--k", type=int, choices=[2, 3, 4], default=3,
        help="Número de grupos; valor por defecto 3, como en la Guía 4")
    parser.add_argument("--salida", type=str, help="Carpeta opcional para los CSV y el modelo")
    args = parser.parse_args()
    archivo = obtener_ruta(args.entrada)
    salida = Path(args.salida).expanduser().resolve() if args.salida else DIRECTORIO_SALIDA
    base = cargar_datos(archivo)
    resultados, evaluacion, perfiles, anual, estabilidad, metadatos, objetos = construir_modelo(base, args.k)
    metadatos["sha256_archivo_origen"] = hashlib.sha256(archivo.read_bytes()).hexdigest()
    exportar(resultados, evaluacion, perfiles, anual, estabilidad, metadatos, objetos, salida)
    print("\nMODELO GMM | REPORTE DE EJECUCIÓN")
    print(f"Fuente: {archivo.name}")
    print(f"Observaciones: {metadatos['observaciones']} | Municipios: {metadatos['municipios']}")
    print(f"PCA (PC1+PC2): {metadatos['varianza_pca2_pct']:.2f} %")
    print(f"GMM k={args.k} | Silueta: {metadatos['silueta']:.3f} | BIC: {metadatos['bic']:.2f}")
    print(f"Estabilidad mínima ARI (4 semillas): {metadatos['ari_minimo_semillas']:.3f}")
    print("\nEvaluación comparativa:")
    print(evaluacion.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nDistribución de los grupos:")
    print(resultados["cluster_gmm"].value_counts().sort_index().to_string())
    print(f"\nArchivos exportados en: {salida}")
    print("Interpretación: los grupos son perfiles exploratorios, no prioridades de inversión.")


if __name__ == "__main__":
    main()
