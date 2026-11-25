"""
3_pca_gmm.py - PCA y K-Means vs GMM

Entrada : datos/Summary municipio-anio 2021-2023.xlsx  (48 filas = 16 municipios x 3 años)
Salidas : resultados/pca_gmm/
            01_pca_varianza.csv          varianza explicada por componente
            02_pca_cargas.csv            cargas (loadings) de cada variable
            03_comparacion_modelos.csv   K-Means y GMM para k = 2, 3, 4 (silueta, Davies-Bouldin, BIC, AIC)
            04_asignacion_gmm.csv        grupo de cada municipio-año y su probabilidad de pertenencia
            05_perfil_grupos.csv         promedio de cada variable por grupo
            06_estabilidad.csv           ARI entre semillas y con remuestreo (bootstrap)
            07_validacion_temporal.csv   modelo ajustado con 2021-2022 y aplicado a 2023
            08_resumen.txt               resumen
            fig_scree.png, fig_plano_pca_gmm.png, fig_metricas_k.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuración 
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
DATOS = REPO / "datos" / "Summary municipio-anio 2021-2023.xlsx"
SALIDA = REPO / "resultados" / "pca_gmm"

VARIABLES = [
    "registros_x1000hab",
    "dinamica_x1000hab",
    "pct_dinamica",
    "licencias_vivienda_x1000hab",
    "unidades_vivienda_x1000hab",
    "unidades_obra_nueva_x1000hab",
]

N_COMPONENTES = 2
K_EVALUADOS = [2, 3, 4]
K_ELEGIDO = 3
SEMILLA = 42
GMM_PARAMS = dict(covariance_type="full", n_init=5, random_state=SEMILLA)
KMEANS_PARAMS = dict(n_init=10, random_state=SEMILLA)
N_SEMILLAS = 30
N_BOOTSTRAP = 200
UMBRAL_TRANSICION = 0.80   # probabilidad < 0,80 = observación "en transición"


def describir_grupos(df_var, etiquetas, orden):
    """Describe cada grupo comparándolo con los demás.

    - El grupo con menor PC1 = menor actividad.
    - De los restantes, el de más registros por habitante = mayor actividad registral.
    - El de más licencias por habitante = mayor licenciamiento de vivienda.
    """
    medias = df_var.groupby(etiquetas).mean()
    desc = {orden[-1]: "Menor actividad registral y constructiva"}
    resto = list(orden[:-1])
    if resto:
        g_reg = medias.loc[resto, "registros_x1000hab"].idxmax()
        desc[g_reg] = "Mayor actividad registral"
        for g in resto:
            if g == g_reg:
                continue
            if medias.loc[g, "licencias_vivienda_x1000hab"] == medias["licencias_vivienda_x1000hab"].max():
                desc[g] = "Mayor licenciamiento de vivienda"
            else:
                desc[g] = "Actividad intermedia"
    return desc


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(DATOS)
    print(f"Datos: {len(df)} filas, {df['municipio'].nunique()} municipios, años {sorted(df['anio'].unique())}")

    # -----------------------------------------------------------------------
    # 1. PCA
    # -----------------------------------------------------------------------
    escalador = StandardScaler()
    Z = escalador.fit_transform(df[VARIABLES])
    pca_total = PCA().fit(Z)

    varianza = pd.DataFrame({
        "componente": [f"PC{i+1}" for i in range(len(VARIABLES))],
        "varianza_pct": pca_total.explained_variance_ratio_ * 100,
        "acumulada_pct": np.cumsum(pca_total.explained_variance_ratio_) * 100,
        "autovalor": pca_total.explained_variance_,
    }).round(2)
    varianza.to_csv(SALIDA / "01_pca_varianza.csv", index=False)

    cargas = pd.DataFrame(pca_total.components_[:N_COMPONENTES].T,
                          index=VARIABLES,
                          columns=[f"PC{i+1}" for i in range(N_COMPONENTES)]).round(3)
    cargas.to_csv(SALIDA / "02_pca_cargas.csv")

    S = pca_total.transform(Z)[:, :N_COMPONENTES]
    df["PC1"], df["PC2"] = S[:, 0].round(3), S[:, 1].round(3)

    # -----------------------------------------------------------------------
    # 2. Comparación K-Means vs GMM
    # -----------------------------------------------------------------------
    filas = []
    for k in K_EVALUADOS:
        km = KMeans(n_clusters=k, **KMEANS_PARAMS).fit(S)
        gmm = GaussianMixture(n_components=k, **GMM_PARAMS).fit(S)
        et_gmm = gmm.predict(S)
        filas.append({
            "k": k,
            "silueta_kmeans": silhouette_score(S, km.labels_),
            "silueta_gmm": silhouette_score(S, et_gmm),
            "davies_bouldin_kmeans": davies_bouldin_score(S, km.labels_),
            "davies_bouldin_gmm": davies_bouldin_score(S, et_gmm),
            "bic_gmm": gmm.bic(S),
            "aic_gmm": gmm.aic(S),
            "prob_media_pertenencia_gmm": gmm.predict_proba(S).max(axis=1).mean(),
            "ari_kmeans_vs_gmm": adjusted_rand_score(km.labels_, et_gmm),
        })
    comparacion = pd.DataFrame(filas).round(3)
    comparacion.to_csv(SALIDA / "03_comparacion_modelos.csv", index=False)

    # -----------------------------------------------------------------------
    # 3. Modelo elegido: GMM con k = 3
    # -----------------------------------------------------------------------
    gmm = GaussianMixture(n_components=K_ELEGIDO, **GMM_PARAMS).fit(S)
    etiquetas = gmm.predict(S)
    proba = gmm.predict_proba(S)

    # Nombre de cada grupo: letra por orden de PC1 (A = mayor) + descripción por perfil
    orden = pd.Series(S[:, 0]).groupby(etiquetas).mean().sort_values(ascending=False).index
    letra = {g: chr(65 + i) for i, g in enumerate(orden)}
    descripcion = describir_grupos(df[VARIABLES], etiquetas, list(orden))
    df["grupo"] = [f"Grupo {letra[g]}" for g in etiquetas]
    df["perfil"] = [descripcion[g] for g in etiquetas]
    df["prob_pertenencia"] = proba.max(axis=1).round(3)
    df["en_transicion"] = df["prob_pertenencia"] < UMBRAL_TRANSICION
    for g in orden:
        df[f"prob_grupo_{letra[g]}"] = proba[:, g].round(3)

    # Atípicos multivariados: 5 % de observaciones con menor densidad en el GMM
    densidad = gmm.score_samples(S)
    df["log_densidad"] = densidad.round(3)
    df["atipico_multivariado"] = densidad <= np.percentile(densidad, 5)

    columnas = (["municipio", "anio", "grupo", "perfil", "prob_pertenencia", "en_transicion"]
                + [f"prob_grupo_{letra[g]}" for g in orden]
                + ["PC1", "PC2", "log_densidad", "atipico_multivariado"])
    asignacion = df[columnas].sort_values(["grupo", "municipio", "anio"])
    asignacion.to_csv(SALIDA / "04_asignacion_gmm.csv", index=False)

    perfil = df.groupby("grupo")[VARIABLES].mean().round(2)
    perfil["n_observaciones"] = df["grupo"].value_counts()
    perfil["perfil"] = df.groupby("grupo")["perfil"].first()
    perfil.to_csv(SALIDA / "05_perfil_grupos.csv")

    # -----------------------------------------------------------------------
    # 4. Estabilidad
    # -----------------------------------------------------------------------
    ari_semillas = [
        adjusted_rand_score(etiquetas, GaussianMixture(K_ELEGIDO, covariance_type="full",
                                                       n_init=5, random_state=s).fit(S).predict(S))
        for s in range(N_SEMILLAS)
    ]
    rng = np.random.default_rng(SEMILLA)
    ari_boot = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(S), len(S), replace=True)
        try:
            g_b = GaussianMixture(K_ELEGIDO, covariance_type="full", n_init=5,
                                  random_state=int(rng.integers(1_000_000)),
                                  reg_covar=1e-4).fit(S[idx])
            ari_boot.append(adjusted_rand_score(etiquetas, g_b.predict(S)))
        except ValueError:
            continue
    estabilidad = pd.DataFrame([
        {"prueba": f"ARI entre {N_SEMILLAS} semillas", "minimo": min(ari_semillas),
         "media": np.mean(ari_semillas), "percentil_5": np.percentile(ari_semillas, 5)},
        {"prueba": f"ARI bootstrap ({len(ari_boot)} remuestreos)", "minimo": min(ari_boot),
         "media": np.mean(ari_boot), "percentil_5": np.percentile(ari_boot, 5)},
    ]).round(3)
    estabilidad.to_csv(SALIDA / "06_estabilidad.csv", index=False)

    # -----------------------------------------------------------------------
    # 5. Validación temporal: ajustar con 2021-2022, aplicar a 2023
    # -----------------------------------------------------------------------
    ultimo = df["anio"].max()
    entreno, prueba = df[df["anio"] < ultimo], df[df["anio"] == ultimo]
    esc_t = StandardScaler().fit(entreno[VARIABLES])
    pca_t = PCA(N_COMPONENTES).fit(esc_t.transform(entreno[VARIABLES]))
    S_e = pca_t.transform(esc_t.transform(entreno[VARIABLES]))
    S_p = pca_t.transform(esc_t.transform(prueba[VARIABLES]))
    gmm_t = GaussianMixture(K_ELEGIDO, **GMM_PARAMS).fit(S_e)
    et_p = gmm_t.predict(S_p)
    validacion = pd.DataFrame([{
        "log_verosimilitud_media_entreno": gmm_t.score(S_e),
        "log_verosimilitud_media_2023": gmm_t.score(S_p),
        "silueta_2023": silhouette_score(S_p, et_p) if len(set(et_p)) > 1 else np.nan,
        "ari_2023_vs_modelo_completo": adjusted_rand_score(etiquetas[df["anio"] == ultimo], et_p),
    }]).round(3)
    validacion.to_csv(SALIDA / "07_validacion_temporal.csv", index=False)

    # -----------------------------------------------------------------------
    # 6. Gráficos
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(varianza["componente"], varianza["varianza_pct"], color="#4C72B0", label="Varianza")
    ax.plot(varianza["componente"], varianza["acumulada_pct"], "o-", color="#C44E52", label="Acumulada")
    ax.set_ylabel("% de varianza")
    ax.set_title("PCA - varianza explicada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_scree.png", dpi=150)
    plt.close(fig)

    colores = ["#C44E52", "#DD8452", "#4C72B0", "#55A868"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for i, g in enumerate(orden):
        m = etiquetas == g
        ax.scatter(S[m, 0], S[m, 1], s=55, color=colores[i], alpha=0.85,
                   label=f"Grupo {letra[g]}: {descripcion[g]} (n={m.sum()})")
    for x, y, mun, an in zip(S[:, 0], S[:, 1], df["municipio"], df["anio"]):
        ax.annotate(f"{mun} {str(an)[-2:]}", (x, y), fontsize=7, alpha=0.75,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(f"PC1 ({varianza.loc[0, 'varianza_pct']:.1f} %) - intensidad registral y de construcción")
    ax.set_ylabel(f"PC2 ({varianza.loc[1, 'varianza_pct']:.1f} %) - proporción de dinámica y licencias")
    ax.set_title("Perfiles de municipios-año (GMM, k = 3)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_plano_pca_gmm.png", dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(comparacion["k"], comparacion["silueta_kmeans"], "o-", label="Silueta K-Means")
    ax1.plot(comparacion["k"], comparacion["silueta_gmm"], "s-", label="Silueta GMM")
    ax1.set_xlabel("Número de grupos (k)")
    ax1.set_ylabel("Silueta")
    ax1.set_xticks(K_EVALUADOS)
    ax2 = ax1.twinx()
    ax2.plot(comparacion["k"], comparacion["bic_gmm"], "^--", color="gray", label="BIC GMM")
    ax2.set_ylabel("BIC (menor es mejor)")
    ax1.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.set_title("Métricas por número de grupos")
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_metricas_k.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 7. Resumen
    # -----------------------------------------------------------------------
    lineas = [
        "RESUMEN PCA + GMM",
        "=" * 60,
        f"Observaciones: {len(df)} | Variables: {len(VARIABLES)}",
        "",
        "PCA:",
        varianza.to_string(index=False),
        f"Criterio de Kaiser (autovalor > 1): {(pca_total.explained_variance_ > 1).sum()} componentes",
        "",
        "Cargas:",
        cargas.to_string(),
        "",
        "Comparación de modelos:",
        comparacion.to_string(index=False),
        "",
        f"Modelo elegido: GMM k = {K_ELEGIDO}, {GMM_PARAMS}",
        "",
        "Perfil de los grupos:",
        perfil.to_string(),
        "",
        "Municipios-año por grupo:",
    ]
    for g in [f"Grupo {letra[x]}" for x in orden]:
        sub = asignacion[asignacion["grupo"] == g]
        lineas.append(f"  {g}: " + ", ".join(f"{m} {a}" for m, a in zip(sub["municipio"], sub["anio"])))
    lineas += [
        "",
        f"En transición (probabilidad < {UMBRAL_TRANSICION}):",
        asignacion[asignacion["en_transicion"]][["municipio", "anio", "grupo", "prob_pertenencia"]].to_string(index=False)
        if asignacion["en_transicion"].any() else "  ninguno",
        "",
        "Atípicos multivariados (5 % de menor densidad):",
        asignacion[asignacion["atipico_multivariado"]][["municipio", "anio", "grupo", "log_densidad"]].to_string(index=False),
        "",
        "Estabilidad:",
        estabilidad.to_string(index=False),
        "",
        "Validación temporal (ajuste 2021-2022, aplicado a 2023):",
        validacion.to_string(index=False),
    ]
    texto = "\n".join(lineas)
    (SALIDA / "08_resumen.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nArchivos guardados en: {SALIDA}")


if __name__ == "__main__":
    main()