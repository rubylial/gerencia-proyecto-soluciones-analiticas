"""
5_iqr_atipicos.py - Detección de valores atípicos en compraventas con el rango intercuartílico (IQR)


Entrada : datos/IGAC transacciones 2021-2023 limpio.parquet
Salidas : resultados/iqr/
            01_escenarios.csv                 registros evaluados, atípicos y proporción en los 4 escenarios
            02_impacto_estadisticos.csv       cómo cambian media y mediana al excluir los atípicos
            03_limites_municipio_anio.csv     Q1, Q3, límites y atípicos por municipio-año (escenario principal)
            04_registros_atipicos.csv         registros marcados como atípicos (escenario principal)
            05_resumen.txt                    resumen legible
            fig_proporcion_atipicos.png, fig_boxplot_valores.png

Reglas (las del documento):
  - Solo compraventas (código de naturaleza jurídica 125) con valor válido (> 0).
  - Límites IQR calculados por municipio y año: [Q1 - 1,5*IQR ; Q3 + 1,5*IQR].
  - Solo se evalúan municipio-año con al menos 30 registros.
  - "Folio único": documentos asociados a una sola matrícula (FOLIOS_EN_DOCUMENTO = 1).
  - Escenario principal: 10 municipios del Valle de Aburrá, folio único.
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IGAC = REPO / "datos" / "IGAC transacciones 2021-2023 limpio.parquet"
SALIDA = REPO / "resultados" / "iqr"

VALLE_ABURRA = ["05001", "05079", "05088", "05129", "05212",
                "05266", "05308", "05360", "05380", "05631"]
COD_COMPRAVENTA = "125"
FACTOR_IQR = 1.5
MIN_OBS = 30
GRUPO = ["DIVIPOLA", "MUNICIPIO", "YEAR_RADICA"]

ESCENARIOS = {
    "10 municipios, folio único": dict(solo_valle=True, folio_unico=True),
    "10 municipios, todos los folios": dict(solo_valle=True, folio_unico=False),
    "16 municipios, folio único": dict(solo_valle=False, folio_unico=True),
    "16 municipios, todos los folios": dict(solo_valle=False, folio_unico=False),
}
PRINCIPAL = "10 municipios, folio único"


def cargar():
    cols = ["PK", "MATRICULA", "DOCUMENTO_JUSTIFICATIVO", "DIVIPOLA", "MUNICIPIO", "YEAR_RADICA",
            "COD_NATUJUR", "NOMBRE_NATUJUR", "VALOR", "VALOR_VALIDO", "FOLIOS_EN_DOCUMENTO"]
    df = pd.read_parquet(IGAC, columns=cols)
    return df[(df["COD_NATUJUR"] == COD_COMPRAVENTA) & df["VALOR_VALIDO"]].copy()


def aplicar_iqr(df):
    """Marca atípicos por municipio-año. Devuelve (registros evaluados, límites por grupo)."""
    n = df.groupby(GRUPO)["VALOR"].transform("size")
    excluidos = df[n < MIN_OBS].groupby(GRUPO).size()
    df = df[n >= MIN_OBS].copy()
    lim = df.groupby(GRUPO)["VALOR"].quantile([0.25, 0.75]).unstack()
    lim.columns = ["Q1", "Q3"]
    lim["IQR"] = lim["Q3"] - lim["Q1"]
    lim["limite_inferior"] = lim["Q1"] - FACTOR_IQR * lim["IQR"]
    lim["limite_superior"] = lim["Q3"] + FACTOR_IQR * lim["IQR"]
    df = df.join(lim, on=GRUPO)
    df["atipico"] = (df["VALOR"] < df["limite_inferior"]) | (df["VALOR"] > df["limite_superior"])
    df["tipo_atipico"] = pd.NA
    df.loc[df["VALOR"] < df["limite_inferior"], "tipo_atipico"] = "bajo"
    df.loc[df["VALOR"] > df["limite_superior"], "tipo_atipico"] = "alto"
    resumen = df.groupby(GRUPO).agg(registros=("VALOR", "size"),
                                    atipicos=("atipico", "sum"),
                                    mediana_valor=("VALOR", "median"))
    lim = lim.join(resumen)
    lim["pct_atipicos"] = lim["atipicos"] / lim["registros"] * 100
    return df, lim, excluidos


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    base = cargar()
    print(f"Compraventas con valor válido (16 municipios, 2021-2023): {len(base):,}")

    filas_esc, filas_imp, resultados = [], [], {}
    for nombre, cfg in ESCENARIOS.items():
        d = base
        if cfg["solo_valle"]:
            d = d[d["DIVIPOLA"].isin(VALLE_ABURRA)]
        if cfg["folio_unico"]:
            d = d[d["FOLIOS_EN_DOCUMENTO"] == 1]
        evaluados, limites, excluidos = aplicar_iqr(d)
        resultados[nombre] = (evaluados, limites)
        n_at = int(evaluados["atipico"].sum())
        filas_esc.append({
            "escenario": nombre,
            "registros_evaluados": len(evaluados),
            "atipicos": n_at,
            "proporcion_pct": round(n_at / len(evaluados) * 100, 2),
            "atipicos_altos": int((evaluados["tipo_atipico"] == "alto").sum()),
            "atipicos_bajos": int((evaluados["tipo_atipico"] == "bajo").sum()),
            "municipio_anio_excluidos_por_menos_de_30": len(excluidos),
        })
        v, sin = evaluados["VALOR"], evaluados.loc[~evaluados["atipico"], "VALOR"]
        filas_imp.append({
            "escenario": nombre,
            "media_con_atipicos": v.mean(),
            "media_sin_atipicos": sin.mean(),
            "cambio_media_pct": (sin.mean() / v.mean() - 1) * 100,
            "mediana_con_atipicos": v.median(),
            "mediana_sin_atipicos": sin.median(),
            "cambio_mediana_pct": (sin.median() / v.median() - 1) * 100,
        })

    escenarios = pd.DataFrame(filas_esc)
    impacto = pd.DataFrame(filas_imp)
    escenarios.to_csv(SALIDA / "01_escenarios.csv", index=False)
    impacto.round(2).to_csv(SALIDA / "02_impacto_estadisticos.csv", index=False)

    evaluados, limites = resultados[PRINCIPAL]
    limites.round(2).reset_index().to_csv(SALIDA / "03_limites_municipio_anio.csv", index=False)
    cols_at = ["PK", "MATRICULA", "DOCUMENTO_JUSTIFICATIVO", "MUNICIPIO", "YEAR_RADICA",
               "VALOR", "limite_inferior", "limite_superior", "tipo_atipico"]
    evaluados.loc[evaluados["atipico"], cols_at].sort_values(["MUNICIPIO", "YEAR_RADICA", "VALOR"]) \
        .to_csv(SALIDA / "04_registros_atipicos.csv", index=False)

    # -----------------------------------------------------------------------
    # Gráficos
    # -----------------------------------------------------------------------
    tabla = limites.reset_index().pivot_table(index="MUNICIPIO", columns="YEAR_RADICA", values="pct_atipicos")
    tabla = tabla.loc[tabla.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tabla.plot.barh(ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_xlabel("% de compraventas marcadas como atípicas")
    ax.set_ylabel("")
    ax.set_title(f"Proporción de atípicos por municipio y año ({PRINCIPAL})")
    ax.legend(title="Año")
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_proporcion_atipicos.png", dpi=150)
    plt.close(fig)

    orden = evaluados.groupby("MUNICIPIO")["VALOR"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([evaluados.loc[evaluados["MUNICIPIO"] == m, "VALOR"] / 1e6 for m in orden],
               vert=False, whis=FACTOR_IQR, flierprops=dict(markersize=2, alpha=0.3))
    ax.set_yticks(range(1, len(orden) + 1))
    ax.set_yticklabels(orden)
    ax.set_xscale("log")
    ax.set_xlabel("Valor registrado (millones de pesos, escala logarítmica)")
    ax.set_title(f"Distribución de valores de compraventa 2021-2023 ({PRINCIPAL})")
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_boxplot_valores.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Resumen
    # -----------------------------------------------------------------------
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    texto = "\n".join([
        "RESUMEN DETECCIÓN DE ATÍPICOS (IQR)",
        "=" * 60,
        f"Reglas: compraventas (código {COD_COMPRAVENTA}) con valor > 0; límites por municipio-año;",
        f"factor {FACTOR_IQR}; mínimo {MIN_OBS} registros por municipio-año.",
        "",
        "Escenarios:",
        escenarios.to_string(index=False),
        "",
        "Impacto de excluir temporalmente los atípicos:",
        impacto.to_string(index=False),
        "",
        f"Detalle por municipio-año ({PRINCIPAL}):",
        limites[["registros", "atipicos", "pct_atipicos", "mediana_valor",
                 "limite_inferior", "limite_superior"]].to_string(),
        "",
        "Nota: un atípico es una observación para revisar, no una prueba de error o irregularidad.",
        "El valor registrado no equivale al precio comercial del inmueble.",
    ])
    (SALIDA / "05_resumen.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nArchivos guardados en: {SALIDA}")


if __name__ == "__main__":
    main()