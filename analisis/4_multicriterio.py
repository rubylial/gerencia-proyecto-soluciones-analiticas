"""
4_multicriterio.py - Preselección de municipios del Valle de Aburrá con suma ponderada
                     y análisis de sensibilidad de las ponderaciones

Entradas : datos/Summary municipio-anio 2021-2023.xlsx           (población y licencias)
           datos/IGAC transacciones 2021-2023 limpio.parquet     (registros de compraventa)
Salidas  : resultados/multicriterio/
            01_indicadores.csv            indicadores originales y normalizados (Min-Max)
            02_ranking_pesos_iguales.csv  puntaje y posición con pesos de 1/3 cada uno
            03_escenarios.csv             66 combinaciones de pesos, puntajes y top 3 de cada una
            04_frecuencia_top3.csv        cuántas veces queda cada municipio entre los 3 primeros
            05_resumen.txt                resumen legible
            fig_ranking.png, fig_frecuencia_top3.png

"""

from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración 
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "datos" / "Summary municipio-anio 2021-2023.xlsx"
IGAC = REPO / "datos" / "IGAC transacciones 2021-2023 limpio.parquet"
SALIDA = REPO / "resultados" / "multicriterio"

VALLE_ABURRA = ["05001", "05079", "05088", "05129", "05212",
                "05266", "05308", "05360", "05380", "05631"]
COD_COMPRAVENTA = "125"          # código de naturaleza jurídica "COMPRAVENTA"
ANIO_INICIAL, ANIO_FINAL = 2021, 2023
PASO_PESOS = 10                  # incrementos de 10 puntos porcentuales -> 66 combinaciones

CRITERIOS = {
    "compraventas_x1000hab": "Compraventas por 1.000 hab.",
    "obra_nueva_x1000hab": "Unidades de vivienda nueva autorizadas por 1.000 hab.",
    "crecimiento_poblacion_pct": "Crecimiento de la población 2021-2023 (%)",
}


def construir_indicadores():
    summary = pd.read_excel(SUMMARY, dtype={"divipola": str})
    summary["divipola"] = summary["divipola"].str.zfill(5)
    summary = summary[summary["divipola"].isin(VALLE_ABURRA)]

    igac = pd.read_parquet(IGAC, columns=["DIVIPOLA", "COD_NATUJUR"])
    compraventas = (igac[igac["DIVIPOLA"].isin(VALLE_ABURRA) & (igac["COD_NATUJUR"] == COD_COMPRAVENTA)]
                    .groupby("DIVIPOLA").size())

    por_mun = summary.groupby("divipola").agg(
        municipio=("municipio", "first"),
        poblacion_acumulada=("poblacion", "sum"),
        obra_nueva=("unidades_obra_nueva", "sum"),
    )
    pob_ini = summary[summary["anio"] == ANIO_INICIAL].set_index("divipola")["poblacion"]
    pob_fin = summary[summary["anio"] == ANIO_FINAL].set_index("divipola")["poblacion"]

    ind = pd.DataFrame({
        "municipio": por_mun["municipio"],
        "compraventas": compraventas.reindex(por_mun.index).fillna(0).astype(int),
        "obra_nueva": por_mun["obra_nueva"],
        "poblacion_acumulada": por_mun["poblacion_acumulada"],
    })
    # Tasas: total del periodo / suma de las poblaciones anuales x 1.000
    ind["compraventas_x1000hab"] = ind["compraventas"] / ind["poblacion_acumulada"] * 1000
    ind["obra_nueva_x1000hab"] = ind["obra_nueva"] / ind["poblacion_acumulada"] * 1000
    ind["crecimiento_poblacion_pct"] = (pob_fin / pob_ini - 1) * 100

    # Normalización Min-Max entre los 10 municipios (0 = mínimo, 1 = máximo)
    for c in CRITERIOS:
        rango = ind[c].max() - ind[c].min()
        ind[f"{c}_norm"] = (ind[c] - ind[c].min()) / rango if rango else 0.0
    return ind.reset_index()


def puntuar(ind, pesos):
    total = sum(pesos.values())
    return sum(ind[f"{c}_norm"] * (w / total) for c, w in pesos.items())


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    ind = construir_indicadores()
    columnas_ind = (["divipola", "municipio", "compraventas", "obra_nueva", "poblacion_acumulada"]
                    + list(CRITERIOS) + [f"{c}_norm" for c in CRITERIOS])
    ind[columnas_ind].round(4).to_csv(SALIDA / "01_indicadores.csv", index=False)

    # -----------------------------------------------------------------------
    # 1. Escenario de referencia: pesos iguales
    # -----------------------------------------------------------------------
    ranking = ind[["municipio"]].copy()
    ranking["puntaje"] = puntuar(ind, {c: 1 for c in CRITERIOS})
    ranking = ranking.sort_values("puntaje", ascending=False).reset_index(drop=True)
    ranking.index += 1
    ranking.index.name = "posicion"
    ranking.round(3).to_csv(SALIDA / "02_ranking_pesos_iguales.csv")

    # -----------------------------------------------------------------------
    # 2. Sensibilidad: todas las combinaciones de pesos con paso de 10 % que suman 100 %
    # -----------------------------------------------------------------------
    niveles = range(0, 101, PASO_PESOS)
    combinaciones = [(a, b, c) for a, b, c in product(niveles, repeat=3) if a + b + c == 100]
    filas = []
    for a, b, c in combinaciones:
        pesos = dict(zip(CRITERIOS, (a, b, c)))
        p = puntuar(ind, pesos)
        orden = ind.assign(puntaje=p).sort_values("puntaje", ascending=False)["municipio"].tolist()
        fila = {f"peso_{k}": v for k, v in pesos.items()}
        fila.update({"primero": orden[0], "segundo": orden[1], "tercero": orden[2]})
        fila.update({f"puntaje_{m}": round(v, 4) for m, v in zip(ind["municipio"], p)})
        filas.append(fila)
    escenarios = pd.DataFrame(filas)
    escenarios.to_csv(SALIDA / "03_escenarios.csv", index=False)

    n = len(escenarios)
    top3 = pd.concat([escenarios["primero"], escenarios["segundo"], escenarios["tercero"]])
    frecuencia = pd.DataFrame({
        "veces_en_top3": top3.value_counts(),
        "veces_primero": escenarios["primero"].value_counts(),
    }).reindex(ind["municipio"]).fillna(0).astype(int)
    frecuencia["pct_en_top3"] = (frecuencia["veces_en_top3"] / n * 100).round(2)
    frecuencia = frecuencia.sort_values(["veces_en_top3", "veces_primero"], ascending=False)
    frecuencia.index.name = "municipio"
    frecuencia.to_csv(SALIDA / "04_frecuencia_top3.csv")

    # Quién gana según el peso del crecimiento poblacional
    w_crec = escenarios["peso_crecimiento_poblacion_pct"]
    bajo = escenarios[w_crec <= 30]["primero"].value_counts()
    alto = escenarios[w_crec >= 50]["primero"].value_counts()

    # -----------------------------------------------------------------------
    # 3. Gráficos
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    r = ranking.iloc[::-1]
    ax.barh(r["municipio"], r["puntaje"], color="#4C72B0")
    for y, v in enumerate(r["puntaje"]):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Puntaje (0 a 1)")
    ax.set_title("Preselección con pesos iguales (1/3 cada criterio)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_ranking.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    f = frecuencia.iloc[::-1]
    ax.barh(f.index, f["pct_en_top3"], color="#55A868")
    for y, v in enumerate(f["pct_en_top3"]):
        ax.text(v + 1, y, f"{v:.1f} %", va="center", fontsize=8)
    ax.set_xlabel(f"% de los {n} escenarios de pesos en que queda entre los 3 primeros")
    ax.set_title("Sensibilidad de la preselección a las ponderaciones")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_frecuencia_top3.png", dpi=150)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 4. Resumen 
    # -----------------------------------------------------------------------
    texto = "\n".join([
        "RESUMEN EVALUACIÓN MULTICRITERIO",
        "=" * 60,
        "Criterios: " + "; ".join(CRITERIOS.values()),
        "Normalización: Min-Max entre los 10 municipios del Valle de Aburrá",
        "",
        "Indicadores originales:",
        ind.set_index("municipio")[list(CRITERIOS)].round(3).to_string(),
        "",
        "Ranking con pesos iguales:",
        ranking.round(3).to_string(),
        "",
        f"Sensibilidad: {n} combinaciones de pesos (paso {PASO_PESOS} %, suma 100 %)",
        frecuencia.to_string(),
        "",
        f"Escenarios con peso del crecimiento <= 30 % ({(w_crec <= 30).sum()}): primer lugar ->",
        bajo.to_string(),
        "",
        f"Escenarios con peso del crecimiento >= 50 % ({(w_crec >= 50).sum()}): primer lugar ->",
        alto.to_string(),
        "",
        "Nota: es una preselección exploratoria, no una recomendación de inversión.",
        "Las ponderaciones son supuestos que la empresa debe validar.",
    ])
    (SALIDA / "05_resumen.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nArchivos guardados en: {SALIDA}")


if __name__ == "__main__":
    main()