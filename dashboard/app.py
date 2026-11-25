import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

# Ruta base = carpeta donde está este script
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Dinámica Inmobiliaria - Dashboard",
    page_icon="🏘️",
    layout="wide",
)

st.title("🏘️ Dinámica Inmobiliaria Municipal")
st.caption("Análisis exploratorio y modelación — Gerencia de Proyectos de Analytics (MIAD)")

# ============================================================
# CARGA DE DATOS
# ============================================================
DATA_PATH = BASE_DIR / ".." / "datos" / "Summary municipio-anio 2021-2023.xlsx"

@st.cache_data
def cargar_datos(path):
    return pd.read_excel(path)

try:
    df = cargar_datos(str(DATA_PATH.resolve()))
except FileNotFoundError:
    st.error(f"No se encontró el archivo: {DATA_PATH.resolve()}")
    st.stop()

if df is None or df.empty:
    st.error("El archivo se cargó vacío. Verifica el Excel.")
    st.stop()

# ============================================================
# DICCIONARIO DE ETIQUETAS LEGIBLES
# ============================================================
ETIQUETAS = {
    "registros": "Registros IGAC",
    "dinamica": "Dinámica inmobiliaria",
    "registros_con_valor": "Registros con valor",
    "licencias_vivienda": "Licencias de vivienda",
    "unidades_vivienda": "Unidades de vivienda",
    "unidades_obra_nueva": "Unidades obra nueva",
    "area_vivienda_m2": "Área vivienda (m²)",
    "registros_x1000hab": "Registros por 1.000 hab.",
    "dinamica_x1000hab": "Dinámica por 1.000 hab.",
    "licencias_vivienda_x1000hab": "Licencias por 1.000 hab.",
    "unidades_vivienda_x1000hab": "Unid. vivienda por 1.000 hab.",
    "unidades_obra_nueva_x1000hab": "Obra nueva por 1.000 hab.",
    "poblacion": "Población",
    "pct_dinamica": "% Dinámica",
    "pct_registros_con_valor": "% Registros con valor",
    "unidades_novis": "Unidades No VIS",
    "unidades_vip": "Unidades VIP",
    "unidades_vis": "Unidades VIS",
    "unidades_apartamentos": "Unidades apartamentos",
    "unidades_casas": "Unidades casas",
}

def etiqueta(var):
    """Devuelve la etiqueta legible de una variable."""
    return ETIQUETAS.get(var, var)

# Variables normalizadas (tasas por 1000 hab) para PCA y clustering
VARS_NORM = [
    "registros_x1000hab",
    "dinamica_x1000hab",
    "licencias_vivienda_x1000hab",
    "unidades_vivienda_x1000hab",
    "unidades_obra_nueva_x1000hab",
]

VARS_ABSOLUTAS = [
    "registros",
    "dinamica",
    "registros_con_valor",
    "licencias_vivienda",
    "unidades_vivienda",
    "unidades_obra_nueva",
    "area_vivienda_m2",
]

# Coordenadas aproximadas de municipios colombianos (para mapa)
COORDS_MUNICIPIOS = {
    "Medellín": (6.2476, -75.5658), "Bello": (6.3384, -75.5586),
    "Envigado": (6.1711, -75.5906), "Itagüí": (6.1849, -75.6063),
    "Sabaneta": (6.1517, -75.6164), "La Estrella": (6.1593, -75.6425),
    "Caldas": (6.0914, -75.6367), "Copacabana": (6.3493, -75.5115),
    "Girardota": (6.3781, -75.4544), "Barbosa": (6.4381, -75.3315),
    "Bogotá": (4.7110, -74.0721), "Bogotá D.C.": (4.7110, -74.0721),
    "Rionegro": (6.1553, -75.3743), "Chía": (4.8637, -74.0540),
    "Cajicá": (4.9189, -74.0282), "Zipaquirá": (5.0225, -73.9937),
    "Soacha": (4.5793, -74.2170), "Funza": (4.7175, -74.2117),
    "Madrid": (4.7345, -74.2644), "Mosquera": (4.6815, -74.2328),
    "Cota": (4.8135, -74.0998), "Tocancipá": (4.9644, -73.9159),
    "Sopó": (4.9064, -73.9398), "La Calera": (4.7210, -73.9679),
    "Cali": (3.4516, -76.5320), "Bucaramanga": (7.1194, -73.1227),
    "Barranquilla": (10.9685, -74.7813), "Cartagena": (10.3910, -75.5143),
    "Pereira": (4.8133, -75.6961), "Manizales": (5.0689, -75.5174),
    "Armenia": (4.5339, -75.6811), "Ibagué": (4.4389, -75.2322),
    "Villavicencio": (4.1420, -73.6266), "Neiva": (2.9273, -75.2819),
    "Santa Marta": (11.2408, -74.1990), "Pasto": (1.2136, -77.2811),
    "Tunja": (5.5353, -73.3678), "Popayán": (2.4419, -76.6063),
    "Floridablanca": (7.0642, -73.0863), "Piedecuesta": (6.9876, -73.0523),
    "Dosquebradas": (4.8392, -75.6723), "Palmira": (3.5394, -76.3036),
}

# ============================================================
# SIDEBAR - FILTROS
# ============================================================
st.sidebar.header("🔎 Filtros")

anios = sorted(df["anio"].unique())
anio_sel = st.sidebar.multiselect("Año", anios, default=anios)

municipios = sorted(df["municipio"].unique())
mun_sel = st.sidebar.multiselect("Municipio", municipios, default=municipios)

df_filtrado = df[df["anio"].isin(anio_sel) & df["municipio"].isin(mun_sel)]

st.sidebar.markdown("---")
st.sidebar.metric("Filas filtradas", len(df_filtrado))
st.sidebar.metric("Municipios", df_filtrado["municipio"].nunique())

# Botón de descarga en sidebar
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="⬇️ Descargar datos filtrados (CSV)",
    data=df_filtrado.to_csv(index=False).encode("utf-8"),
    file_name="datos_filtrados.csv",
    mime="text/csv",
)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def calcular_crecimiento(df_base):
    """Calcula tasa de crecimiento interanual entre primer y último año."""
    anios_disp = sorted(df_base["anio"].unique())
    if len(anios_disp) < 2:
        return None
    anio_ini, anio_fin = anios_disp[0], anios_disp[-1]
    n_anios = anio_fin - anio_ini
    if n_anios == 0:
        return None

    df_ini = df_base[df_base["anio"] == anio_ini].set_index("municipio")
    df_fin = df_base[df_base["anio"] == anio_fin].set_index("municipio")

    municipios_comunes = df_ini.index.intersection(df_fin.index)
    if len(municipios_comunes) == 0:
        return None

    resultados = []
    for mun in municipios_comunes:
        fila = {"municipio": mun}
        for var in VARS_NORM:
            val_ini = df_ini.loc[mun, var]
            val_fin = df_fin.loc[mun, var]
            if val_ini > 0:
                tasa = ((val_fin / val_ini) ** (1 / n_anios) - 1) * 100
            else:
                tasa = np.nan
            fila[var] = round(tasa, 1)
        resultados.append(fila)

    return pd.DataFrame(resultados)

def semaforo(val):
    """Retorna emoji de semáforo según valor de crecimiento."""
    if pd.isna(val):
        return "⚪"
    if val > 5:
        return "🟢"
    elif val > 0:
        return "🟡"
    else:
        return "🔴"


# ============================================================
# RESULTADOS DE LOS SCRIPTS DE ANÁLISIS (carpeta resultados/)
# Generados con: python analisis/3_pca_gmm.py, 4_multicriterio.py, 5_iqr_atipicos.py
# ============================================================
RESULTADOS_DIR = (BASE_DIR / ".." / "resultados").resolve()

SCRIPT_DE = {
    "pca_gmm": "analisis/3_pca_gmm.py",
    "multicriterio": "analisis/4_multicriterio.py",
    "iqr": "analisis/5_iqr_atipicos.py",
}


@st.cache_data
def leer_resultado(carpeta, archivo, **kwargs):
    """Lee un CSV de resultados/<carpeta>/<archivo>. Devuelve None si no existe."""
    ruta = RESULTADOS_DIR / carpeta / archivo
    if not ruta.exists():
        return None
    return pd.read_csv(ruta, **kwargs)


def aviso_sin_resultados(carpeta):
    st.warning(
        f"No se encontraron los resultados en `resultados/{carpeta}/`. "
        f"Ejecuta primero, desde la raíz del repo: `python {SCRIPT_DE[carpeta]}`",
        icon="⚠️",
    )


# Configuración del documento para PCA y agrupamiento
VARS_PCA = [
    "registros_x1000hab",
    "dinamica_x1000hab",
    "pct_dinamica",
    "licencias_vivienda_x1000hab",
    "unidades_vivienda_x1000hab",
    "unidades_obra_nueva_x1000hab",
]

# ============================================================
# ADVERTENCIAS DE INTERPRETACIÓN
# ============================================================
with st.expander("⚠️ Cómo interpretar este tablero (léelo antes de usarlo)", expanded=False):
    st.markdown(
        "- **Es una herramienta de preselección**, no una recomendación de inversión. "
        "Indica qué municipios merecen estudios más detallados.\n"
        "- **El valor registrado no es el precio de mercado** del inmueble: es el valor declarado en el acto jurídico.\n"
        "- **Un valor atípico no es un error ni una irregularidad**: es una observación que conviene revisar.\n"
        "- **Las ponderaciones del multicriterio son supuestos**.\n"
        "- Datos: IGAC (registros 2021-2023), DANE (proyecciones de población y licencias ELIC). "
        "Nivel municipal."
    )

# ============================================================
# NAVEGACIÓN POR PESTAÑAS
# ============================================================
tab_resumen, tab_ranking, tab_ficha, tab_correlaciones, tab_pca, tab_iqr, tab_clusters, tab_modelo = st.tabs([
    "📊 Resumen",
    "🏆 Preselección multicriterio",
    "🔍 Ficha Municipal",
    "📈 Correlaciones",
    "🔬 PCA",
    "📦 Atípicos (IQR)",
    "🎯 K-Means",
    "🧪 Modelo GMM",
])

# ============================================================
# TAB 1 - RESUMEN GENERAL
# ============================================================
with tab_resumen:
    st.subheader("Panorama general")
    st.info(
        "📌 Vista consolidada de los indicadores clave de dinámica inmobiliaria. "
        "Las flechas (Δ) muestran la variación respecto al año anterior.",
        icon="ℹ️",
    )

    # Calcular deltas vs año anterior
    anio_max = df_filtrado["anio"].max() if len(df_filtrado) > 0 else None
    anio_prev = anio_max - 1 if anio_max else None

    metricas_vars = ["registros", "dinamica", "licencias_vivienda", "unidades_vivienda"]
    metricas_labels = [etiqueta(v) for v in metricas_vars]

    cols = st.columns(4)
    for i, (var, label) in enumerate(zip(metricas_vars, metricas_labels)):
        val_actual = df_filtrado[df_filtrado["anio"] == anio_max][var].sum() if anio_max else 0
        val_prev = df_filtrado[df_filtrado["anio"] == anio_prev][var].sum() if anio_prev else None
        delta = None
        if val_prev and val_prev > 0:
            delta = f"{((val_actual / val_prev) - 1) * 100:+.1f}%"
        with cols[i]:
            st.metric(label, f"{val_actual:,.0f}", delta=delta)

    st.markdown("---")

    # Mapa de municipios
    st.subheader("🗺️ Mapa de municipios")
    var_mapa = st.selectbox("Variable para el mapa", VARS_NORM, index=0, key="var_mapa",
                            format_func=etiqueta)
    anio_mapa = st.selectbox("Año", anios, index=len(anios) - 1, key="anio_mapa")
    df_mapa = df_filtrado[df_filtrado["anio"] == anio_mapa].copy()

    # Agregar coordenadas
    df_mapa["lat"] = df_mapa["municipio"].map(lambda m: COORDS_MUNICIPIOS.get(m, (None, None))[0])
    df_mapa["lon"] = df_mapa["municipio"].map(lambda m: COORDS_MUNICIPIOS.get(m, (None, None))[1])
    df_mapa_geo = df_mapa.dropna(subset=["lat", "lon"])

    if len(df_mapa_geo) > 0:
        fig_mapa = px.scatter_map(
            df_mapa_geo,
            lat="lat", lon="lon",
            size=var_mapa, color=var_mapa,
            hover_name="municipio",
            hover_data={var_mapa: ":.2f", "poblacion": ":,.0f", "lat": False, "lon": False},
            color_continuous_scale="YlOrRd",
            size_max=30, zoom=5,
            title=f"{etiqueta(var_mapa)} — {anio_mapa}",
        )
        fig_mapa.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_mapa, use_container_width=True)

        sin_coords = set(df_mapa["municipio"]) - set(df_mapa_geo["municipio"])
        if sin_coords:
            st.caption(f"⚠️ Sin coordenadas: {', '.join(sorted(sin_coords))}")
    else:
        st.warning("No se encontraron coordenadas para los municipios seleccionados.")

    st.markdown("---")

    # Evolución temporal
    st.subheader("Evolución temporal por municipio")
    var_temporal = st.selectbox(
        "Variable", VARS_NORM + VARS_ABSOLUTAS, index=0, key="var_temporal",
        format_func=etiqueta,
    )
    fig_temp = px.line(
        df_filtrado, x="anio", y=var_temporal, color="municipio",
        markers=True, title=f"{etiqueta(var_temporal)} por año",
        labels={"anio": "Año", var_temporal: etiqueta(var_temporal), "municipio": "Municipio"},
    )
    fig_temp.update_layout(height=450)
    st.plotly_chart(fig_temp, use_container_width=True)

    # Comparativo entre municipios
    st.subheader("Comparativo entre municipios")
    var_comparar = st.selectbox("Variable", VARS_NORM, index=0, key="var_comparar",
                                format_func=etiqueta)
    anio_comparar = st.selectbox("Año", anios, index=len(anios) - 1, key="anio_comp")
    df_comp = df_filtrado[df_filtrado["anio"] == anio_comparar].sort_values(var_comparar, ascending=True)
    fig_comp = px.bar(
        df_comp, x=var_comparar, y="municipio", orientation="h",
        title=f"{etiqueta(var_comparar)} — {anio_comparar}",
        color=var_comparar, color_continuous_scale="Viridis",
        labels={var_comparar: etiqueta(var_comparar), "municipio": "Municipio"},
    )
    fig_comp.update_layout(height=max(400, len(df_comp) * 25))
    st.plotly_chart(fig_comp, use_container_width=True)

# ============================================================
# TAB 2 - PRESELECCIÓN MULTICRITERIO
# ============================================================
with tab_ranking:
    st.subheader("🏆 Preselección multicriterio — Valle de Aburrá")
    st.info(
        "📌 Ordena los 10 municipios del Valle de Aburrá con una suma ponderada de tres criterios "
        "normalizados de 0 a 1 (Min-Max). Ajusta los pesos según las prioridades de la empresa. "
        "Es una **preselección para estudios posteriores**, no una recomendación de inversión.",
        icon="ℹ️",
    )

    ind_mc = leer_resultado("multicriterio", "01_indicadores.csv", dtype={"divipola": str})
    frec_mc = leer_resultado("multicriterio", "04_frecuencia_top3.csv")

    if ind_mc is None:
        aviso_sin_resultados("multicriterio")
    else:
        CRITERIOS_MC = {
            "compraventas_x1000hab": "Compraventas por 1.000 hab.",
            "obra_nueva_x1000hab": "Vivienda nueva autorizada por 1.000 hab.",
            "crecimiento_poblacion_pct": "Crecimiento poblacional 2021-2023 (%)",
        }

        st.markdown("#### Pesos de los criterios")
        c1, c2, c3 = st.columns(3)
        pesos = {}
        for col, (crit, nombre) in zip((c1, c2, c3), CRITERIOS_MC.items()):
            with col:
                pesos[crit] = st.slider(nombre, 0, 100, 33, step=1, key=f"peso_{crit}")

        total = sum(pesos.values())
        if total == 0:
            st.warning("Asigna un peso mayor que cero a al menos un criterio.")
        else:
            st.caption("Pesos efectivos (normalizados a 100 %): " + " · ".join(
                f"{CRITERIOS_MC[c]}: {w / total * 100:.0f} %" for c, w in pesos.items()))

            rank_mc = ind_mc[["municipio"] + list(CRITERIOS_MC)].copy()
            rank_mc["puntaje"] = sum(ind_mc[f"{c}_norm"] * (w / total) for c, w in pesos.items())
            rank_mc = rank_mc.sort_values("puntaje", ascending=False).reset_index(drop=True)
            rank_mc.index = rank_mc.index + 1

            col_g, col_t = st.columns([3, 2])
            with col_g:
                fig_mc = px.bar(
                    rank_mc.iloc[::-1], x="puntaje", y="municipio", orientation="h",
                    text=rank_mc.iloc[::-1]["puntaje"].round(3),
                    title="Puntaje multicriterio (0 a 1)",
                    labels={"puntaje": "Puntaje", "municipio": ""},
                    color="puntaje", color_continuous_scale="Blues", range_x=[0, 1],
                )
                fig_mc.update_layout(height=420, coloraxis_showscale=False)
                st.plotly_chart(fig_mc, use_container_width=True)
            with col_t:
                st.markdown("#### Preselección (3 primeros)")
                for pos, fila in rank_mc.head(3).iterrows():
                    st.markdown(f"**{pos}. {fila['municipio']}** — puntaje {fila['puntaje']:.3f}")
                st.markdown("#### Indicadores originales")
                st.dataframe(
                    rank_mc[["municipio"] + list(CRITERIOS_MC)].rename(columns=CRITERIOS_MC).round(2),
                    use_container_width=True,
                )

        if frec_mc is not None:
            st.markdown("---")
            st.markdown("#### ¿Qué tan firme es la preselección? (análisis de sensibilidad)")
            st.caption(
                "Se probaron 66 combinaciones de pesos (pasos de 10 %, suma 100 %). "
                "La barra muestra en qué porcentaje de ellas cada municipio queda entre los 3 primeros."
            )
            fig_sens = px.bar(
                frec_mc.sort_values("pct_en_top3"), x="pct_en_top3", y="municipio", orientation="h",
                text="pct_en_top3", range_x=[0, 100],
                labels={"pct_en_top3": "% de escenarios en el top 3", "municipio": ""},
                color="pct_en_top3", color_continuous_scale="Greens",
            )
            fig_sens.update_traces(texttemplate="%{text:.1f} %")
            fig_sens.update_layout(height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_sens, use_container_width=True)
            st.caption(
                "Nota: las diferencias de crecimiento poblacional entre municipios son pequeñas "
                "(1,74 % a 1,85 %) y la normalización Min-Max amplifica su efecto en el ranking."
            )

# ============================================================
# TAB 3 - FICHA MUNICIPAL
# ============================================================
with tab_ficha:
    st.subheader("🔍 Ficha de municipio")
    st.info(
        "📌 Perfil completo de un municipio: evolución, posición relativa y comparación vs. promedio.",
        icon="ℹ️",
    )

    mun_ficha = st.selectbox("Selecciona un municipio", municipios, key="mun_ficha")
    df_mun = df[df["municipio"] == mun_ficha].sort_values("anio")

    if len(df_mun) > 0:
        ultimo = df_mun[df_mun["anio"] == df_mun["anio"].max()].iloc[0]
        penultimo = df_mun[df_mun["anio"] == df_mun["anio"].max() - 1]

        st.markdown(f"### {mun_ficha}")
        cols_ficha = st.columns(5)
        for i, var in enumerate(VARS_NORM):
            val = ultimo[var]
            delta = None
            if len(penultimo) > 0:
                val_prev = penultimo.iloc[0][var]
                if val_prev > 0:
                    delta = f"{((val / val_prev) - 1) * 100:+.1f}%"
            with cols_ficha[i]:
                short_label = etiqueta(var).replace(" por 1.000 hab.", "\n/1000hab")
                st.metric(short_label, f"{val:,.1f}", delta=delta)

        st.markdown("---")

        col_ev1, col_ev2 = st.columns(2)
        with col_ev1:
            st.markdown("#### Evolución temporal")
            var_ficha = st.selectbox("Variable", VARS_NORM, index=1, key="var_ficha_ev",
                                     format_func=etiqueta)
            fig_ev = px.bar(
                df_mun, x="anio", y=var_ficha,
                title=f"{etiqueta(var_ficha)} — {mun_ficha}",
                labels={"anio": "Año", var_ficha: etiqueta(var_ficha)},
                color=var_ficha, color_continuous_scale="Blues",
            )
            fig_ev.update_layout(height=350)
            st.plotly_chart(fig_ev, use_container_width=True)

        with col_ev2:
            st.markdown("#### Posición relativa (último año)")
            anio_ult = df_mun["anio"].max()
            df_todos_ult = df[df["anio"] == anio_ult]

            percentiles = {}
            for var in VARS_NORM:
                rank = (df_todos_ult[var] <= ultimo[var]).sum()
                percentiles[etiqueta(var)] = round(rank / len(df_todos_ult) * 100, 0)

            df_perc = pd.DataFrame({
                "Variable": percentiles.keys(),
                "Percentil": percentiles.values(),
            })
            fig_perc = px.bar(
                df_perc, x="Percentil", y="Variable", orientation="h",
                title=f"Percentil entre {len(df_todos_ult)} municipios ({anio_ult})",
                range_x=[0, 100],
                color="Percentil", color_continuous_scale="RdYlGn",
            )
            fig_perc.update_layout(height=350)
            st.plotly_chart(fig_perc, use_container_width=True)

        # Radar del municipio vs promedio
        st.markdown("#### Comparación vs. promedio general")
        df_todos_ult = df[df["anio"] == df_mun["anio"].max()]
        promedios = df_todos_ult[VARS_NORM].mean()
        valores_mun = ultimo[VARS_NORM]

        all_vals = pd.concat([promedios, valores_mun])
        rango_r = all_vals.max() - all_vals.min()
        if rango_r == 0:
            rango_r = 1
        prom_norm = ((promedios - all_vals.min()) / rango_r).values.tolist()
        mun_norm = ((valores_mun - all_vals.min()) / rango_r).values.tolist()
        theta_f = [etiqueta(v).replace(" por 1.000 hab.", "") for v in VARS_NORM]

        fig_radar_ficha = go.Figure()
        fig_radar_ficha.add_trace(go.Scatterpolar(
            r=prom_norm + [prom_norm[0]], theta=theta_f + [theta_f[0]],
            name="Promedio general", fill="toself", opacity=0.3,
        ))
        fig_radar_ficha.add_trace(go.Scatterpolar(
            r=mun_norm + [mun_norm[0]], theta=theta_f + [theta_f[0]],
            name=mun_ficha, fill="toself", opacity=0.5,
        ))
        fig_radar_ficha.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400,
        )
        st.plotly_chart(fig_radar_ficha, use_container_width=True)

        # Resultados de los modelos para este municipio
        st.markdown("#### Resultados de los modelos para este municipio")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            asig_f = leer_resultado("pca_gmm", "04_asignacion_gmm.csv")
            if asig_f is not None:
                sub_f = asig_f[asig_f["municipio"] == mun_ficha][["anio", "grupo", "perfil", "prob_pertenencia"]]
                st.markdown("**Perfil GMM por año**")
                st.dataframe(sub_f.rename(columns={"anio": "Año", "grupo": "Grupo", "perfil": "Perfil",
                                                   "prob_pertenencia": "Probabilidad"}),
                             use_container_width=True, hide_index=True)
            else:
                aviso_sin_resultados("pca_gmm")
        with col_m2:
            rk_f = leer_resultado("multicriterio", "02_ranking_pesos_iguales.csv")
            fr_f = leer_resultado("multicriterio", "04_frecuencia_top3.csv")
            if rk_f is not None and mun_ficha in set(rk_f["municipio"]):
                fila_rk = rk_f[rk_f["municipio"] == mun_ficha].iloc[0]
                st.metric("Posición multicriterio (pesos iguales)", f"{int(fila_rk['posicion'])} de {len(rk_f)}",
                          help=f"Puntaje {fila_rk['puntaje']:.3f}")
                if fr_f is not None:
                    pct = fr_f.loc[fr_f["municipio"] == mun_ficha, "pct_en_top3"].iloc[0]
                    st.metric("Escenarios de pesos en el top 3", f"{pct:.1f} %")
            else:
                st.caption("El multicriterio solo incluye los 10 municipios del Valle de Aburrá.")
        st.caption("⚠️ Valores registrados ≠ precios de mercado. La ficha apoya la preselección, "
                   "no reemplaza estudios financieros, comerciales, técnicos ni urbanísticos.")
    else:
        st.warning("No hay datos para este municipio.")

# ============================================================
# TAB 4 - CORRELACIONES
# ============================================================
with tab_correlaciones:
    st.subheader("📈 Mapa de correlaciones")
    st.info(
        "📌 Relación lineal entre variables. Valores cercanos a +1 (azul) = relación directa, "
        "cercanos a -1 (rojo) = relación inversa. Útil para entender qué variables se mueven juntas.",
        icon="ℹ️",
    )

    anio_corr = st.selectbox("Año", anios, index=len(anios) - 1, key="anio_corr")
    vars_corr = st.multiselect(
        "Variables a incluir", VARS_NORM + VARS_ABSOLUTAS,
        default=VARS_NORM, key="vars_corr",
        format_func=etiqueta,
    )

    if len(vars_corr) >= 2:
        df_corr = df[df["anio"] == anio_corr][vars_corr].dropna()
        corr_matrix = df_corr.corr().round(2)

        labels_corr = [etiqueta(v) for v in corr_matrix.columns]

        fig_heat = px.imshow(
            corr_matrix.values,
            x=labels_corr, y=labels_corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=True,
            title=f"Correlación de Pearson — {anio_corr}",
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Scatter de dos variables
        st.markdown("#### Explorar relación entre dos variables")
        col_x, col_y = st.columns(2)
        with col_x:
            var_x = st.selectbox("Eje X", vars_corr, index=0, key="scatter_x",
                                 format_func=etiqueta)
        with col_y:
            var_y = st.selectbox("Eje Y", vars_corr,
                                 index=min(1, len(vars_corr) - 1), key="scatter_y",
                                 format_func=etiqueta)

        df_scatter = df[df["anio"] == anio_corr].dropna(subset=[var_x, var_y])
        fig_scatter = px.scatter(
            df_scatter, x=var_x, y=var_y, hover_name="municipio",
            trendline="ols",
            title=f"{etiqueta(var_x)} vs {etiqueta(var_y)} — {anio_corr}",
            labels={var_x: etiqueta(var_x), var_y: etiqueta(var_y)},
        )
        fig_scatter.update_traces(marker=dict(size=10))
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Selecciona al menos 2 variables.")

# ============================================================
# TAB 5 - PCA (misma configuración del documento: 48 obs., 6 variables)
# ============================================================
with tab_pca:
    st.subheader("Análisis de Componentes Principales (PCA)")
    st.info(
        "📌 Resume 6 indicadores por habitante en 2 componentes, usando las 48 observaciones "
        "municipio-año (16 territorios × 2021-2023). Puntos cercanos = perfiles inmobiliarios similares. "
        "El % de registros con valor se excluye porque mide la cobertura de la fuente, no el mercado.",
        icon="ℹ️",
    )

    df_pca = df.dropna(subset=VARS_PCA).copy()
    escalador_pca = StandardScaler()
    X_pca = escalador_pca.fit_transform(df_pca[VARS_PCA])
    pca = PCA().fit(X_pca)
    componentes = pca.transform(X_pca)
    df_pca["PC1"], df_pca["PC2"] = componentes[:, 0], componentes[:, 1]
    df_pca["etiqueta"] = df_pca["municipio"] + " " + df_pca["anio"].astype(str).str[-2:]

    var_exp = pca.explained_variance_ratio_ * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("Varianza PC1", f"{var_exp[0]:.2f} %")
    m2.metric("Varianza PC2", f"{var_exp[1]:.2f} %")
    m3.metric("PC1 + PC2", f"{var_exp[:2].sum():.2f} %")

    col_scree, col_biplot = st.columns(2)
    with col_scree:
        st.markdown("#### Varianza explicada")
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(var_exp))], y=var_exp, name="Individual"))
        fig_scree.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(var_exp))], y=np.cumsum(var_exp),
                                       name="Acumulada", mode="lines+markers"))
        fig_scree.update_layout(yaxis_title="% Varianza explicada", height=420)
        st.plotly_chart(fig_scree, use_container_width=True)

    with col_biplot:
        st.markdown("#### Biplot (PC1 vs PC2)")
        anios_pca = st.multiselect("Años a mostrar", anios, default=anios, key="anios_pca")
        vista = df_pca[df_pca["anio"].isin(anios_pca)]
        fig_bi = px.scatter(vista, x="PC1", y="PC2", text="etiqueta", color=vista["anio"].astype(str),
                            labels={"color": "Año"})
        cargas_pc = pca.components_[:2, :]
        escala = float(np.abs(componentes[:, :2]).max()) * 0.8
        for i, var in enumerate(VARS_PCA):
            fig_bi.add_annotation(ax=0, ay=0, axref="x", ayref="y",
                                  x=cargas_pc[0, i] * escala, y=cargas_pc[1, i] * escala,
                                  showarrow=True, arrowhead=2, arrowsize=1.5,
                                  text=etiqueta(var).replace(" por 1.000 hab.", ""))
        fig_bi.update_traces(textposition="top center", marker=dict(size=8), textfont_size=9)
        fig_bi.update_layout(height=420)
        st.plotly_chart(fig_bi, use_container_width=True)

    with st.expander("Ver cargas por componente"):
        st.dataframe(pd.DataFrame(pca.components_[:2].T, index=[etiqueta(v) for v in VARS_PCA],
                                  columns=["PC1", "PC2"]).round(3), use_container_width=True)
        st.caption("PC1 ≈ intensidad registral y de construcción · PC2 ≈ proporción de dinámica y licencias.")

# ============================================================
# TAB 6 - ATÍPICOS EN VALORES DE COMPRAVENTA (IQR)
# ============================================================
with tab_iqr:
    st.subheader("📦 Valores atípicos en compraventas (IQR)")
    st.info(
        "📌 Compraventas (código 125) con valor válido. Límites IQR (factor 1,5) calculados por "
        "municipio y año, con mínimo 30 registros. **Un atípico es una observación para revisar, "
        "no un error ni una irregularidad.** El valor registrado no es el precio comercial.",
        icon="ℹ️",
    )

    esc_iqr = leer_resultado("iqr", "01_escenarios.csv")
    imp_iqr = leer_resultado("iqr", "02_impacto_estadisticos.csv")
    lim_iqr = leer_resultado("iqr", "03_limites_municipio_anio.csv", dtype={"DIVIPOLA": str})

    if esc_iqr is None:
        aviso_sin_resultados("iqr")
    else:
        st.markdown("#### Escenarios evaluados")
        tabla_esc = esc_iqr.merge(imp_iqr[["escenario", "cambio_media_pct", "cambio_mediana_pct"]],
                                  on="escenario") if imp_iqr is not None else esc_iqr
        st.dataframe(
            tabla_esc.rename(columns={
                "escenario": "Escenario", "registros_evaluados": "Registros evaluados",
                "atipicos": "Atípicos", "proporcion_pct": "Proporción (%)",
                "atipicos_altos": "Altos", "atipicos_bajos": "Bajos",
                "cambio_media_pct": "Cambio media (%)", "cambio_mediana_pct": "Cambio mediana (%)",
                "municipio_anio_excluidos_por_menos_de_30": "Mun-año excluidos (<30)",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Casi todos los atípicos son valores altos: el límite inferior del IQR suele ser negativo, "
            "así que en la práctica el método marca valores excepcionalmente altos."
        )

        if lim_iqr is not None:
            st.markdown("#### Detalle por municipio y año — escenario 10 municipios, folio único")
            fig_pct = px.bar(
                lim_iqr, x="pct_atipicos", y="MUNICIPIO", color=lim_iqr["YEAR_RADICA"].astype(str),
                barmode="group", orientation="h",
                labels={"pct_atipicos": "% de compraventas atípicas", "MUNICIPIO": "", "color": "Año"},
            )
            fig_pct.update_layout(height=500)
            st.plotly_chart(fig_pct, use_container_width=True)

            vista_lim = lim_iqr[["MUNICIPIO", "YEAR_RADICA", "registros", "atipicos", "pct_atipicos",
                                 "mediana_valor", "limite_superior"]].copy()
            for c in ["mediana_valor", "limite_superior"]:
                vista_lim[c] = (vista_lim[c] / 1e6).round(1)
            st.dataframe(
                vista_lim.rename(columns={
                    "MUNICIPIO": "Municipio", "YEAR_RADICA": "Año", "registros": "Registros",
                    "atipicos": "Atípicos", "pct_atipicos": "% atípicos",
                    "mediana_valor": "Mediana (millones $)", "limite_superior": "Límite superior (millones $)",
                }),
                use_container_width=True, hide_index=True,
            )

# ============================================================
# TAB 7 - K-MEANS (sobre el mismo espacio PCA del documento)
# ============================================================
with tab_clusters:
    st.subheader("🎯 Agrupamiento K-Means (comparación)")
    st.info(
        "📌 K-Means sobre PC1 y PC2 de las 48 observaciones, como referencia para comparar con el "
        "modelo elegido (GMM). Mueve k y observa cómo cambian los grupos y la silueta "
        "(más alta = grupos mejor separados).",
        icon="ℹ️",
    )

    S_km = df_pca[["PC1", "PC2"]].to_numpy()
    n_clusters = st.slider("Número de grupos (k)", 2, 6, 3, key="n_clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(S_km)
    df_km = df_pca[["municipio", "anio", "etiqueta", "PC1", "PC2"] + VARS_PCA].copy()
    df_km["grupo"] = (kmeans.labels_ + 1).astype(str)

    st.metric("Silueta", f"{silhouette_score(S_km, kmeans.labels_):.3f}",
              help="De -1 a 1. Mayor a 0,5 = buena separación.")

    fig_km = px.scatter(df_km, x="PC1", y="PC2", color="grupo", text="etiqueta",
                        hover_data=["municipio", "anio"],
                        title=f"K-Means (k = {n_clusters}) sobre PC1-PC2",
                        color_discrete_sequence=px.colors.qualitative.Set2)
    fig_km.update_traces(textposition="top center", marker=dict(size=10), textfont_size=9)
    fig_km.update_layout(height=500)
    st.plotly_chart(fig_km, use_container_width=True)

    st.markdown("#### Promedios por grupo")
    resumen_km = df_km.groupby("grupo")[VARS_PCA].mean().round(2)
    resumen_km["observaciones"] = df_km["grupo"].value_counts()
    st.dataframe(resumen_km.rename(columns={v: etiqueta(v) for v in VARS_PCA}), use_container_width=True)

    with st.expander("🔎 Silueta por número de grupos"):
        k_range = list(range(2, 7))
        siluetas = [silhouette_score(S_km, KMeans(k, random_state=42, n_init=10).fit_predict(S_km))
                    for k in k_range]
        fig_sil = px.line(x=k_range, y=siluetas, markers=True,
                          labels={"x": "Número de grupos (k)", "y": "Silueta"})
        fig_sil.update_layout(height=300)
        st.plotly_chart(fig_sil, use_container_width=True)

# ============================================================
# TAB 8 - MODELO GMM (modelo elegido)
# ============================================================
with tab_modelo:
    st.subheader("🧪 Modelo de mezclas gaussianas (GMM) — modelo elegido")
    st.info(
        "📌 El GMM agrupa las observaciones municipio-año en perfiles y da a cada una una "
        "**probabilidad de pertenencia**. Los grupos son perfiles estadísticos, no categorías de "
        "atractivo ni de viabilidad de inversión.",
        icon="ℹ️",
    )

    asig = leer_resultado("pca_gmm", "04_asignacion_gmm.csv")
    comp = leer_resultado("pca_gmm", "03_comparacion_modelos.csv")
    perf = leer_resultado("pca_gmm", "05_perfil_grupos.csv")
    estab = leer_resultado("pca_gmm", "06_estabilidad.csv")
    valt = leer_resultado("pca_gmm", "07_validacion_temporal.csv")

    if asig is None:
        aviso_sin_resultados("pca_gmm")
    else:
        asig["etiqueta"] = asig["municipio"] + " " + asig["anio"].astype(str).str[-2:]
        asig["grupo_perfil"] = asig["grupo"] + ": " + asig["perfil"]

        fig_gmm = px.scatter(
            asig, x="PC1", y="PC2", color="grupo_perfil", text="etiqueta",
            size=asig["prob_pertenencia"].clip(lower=0.3), size_max=14,
            hover_data={"municipio": True, "anio": True, "prob_pertenencia": ":.3f",
                        "etiqueta": False, "grupo_perfil": False},
            labels={"grupo_perfil": "Grupo"},
            title="Perfiles de municipios-año (GMM, k = 3)",
            color_discrete_sequence=["#C44E52", "#DD8452", "#4C72B0"],
        )
        fig_gmm.update_traces(textposition="top center", textfont_size=9)
        fig_gmm.update_layout(height=550, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_gmm, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Municipios por grupo")
            for g, sub in asig.groupby("grupo"):
                st.markdown(f"**{g} — {sub['perfil'].iloc[0]}** ({len(sub)} obs.)")
                st.caption(", ".join(f"{m} {a}" for m, a in zip(sub["municipio"], sub["anio"])))
            trans = asig[asig["en_transicion"]]
            if len(trans):
                st.markdown("#### En transición (probabilidad < 0,80)")
                st.dataframe(trans[["municipio", "anio", "grupo", "prob_pertenencia"]],
                             use_container_width=True, hide_index=True)
        with col_b:
            if perf is not None:
                st.markdown("#### Perfil promedio de cada grupo")
                st.dataframe(perf.set_index("grupo").drop(columns=["perfil"])
                             .rename(columns={v: etiqueta(v) for v in VARS_PCA}).T,
                             use_container_width=True)

        st.markdown("---")
        st.markdown("#### Evaluación del modelo")
        if comp is not None:
            st.markdown("**K-Means vs GMM por número de grupos**")
            st.dataframe(comp.rename(columns={
                "k": "k", "silueta_kmeans": "Silueta K-Means", "silueta_gmm": "Silueta GMM",
                "davies_bouldin_kmeans": "Davies-Bouldin K-Means", "davies_bouldin_gmm": "Davies-Bouldin GMM",
                "bic_gmm": "BIC GMM", "aic_gmm": "AIC GMM",
                "prob_media_pertenencia_gmm": "Prob. media pertenencia", "ari_kmeans_vs_gmm": "ARI K-Means vs GMM",
            }), use_container_width=True, hide_index=True)
            st.caption("Silueta: mayor es mejor · Davies-Bouldin y BIC: menor es mejor.")
        c_e, c_v = st.columns(2)
        with c_e:
            if estab is not None:
                st.markdown("**Estabilidad (ARI: 1 = asignaciones idénticas)**")
                st.dataframe(estab, use_container_width=True, hide_index=True)
        with c_v:
            if valt is not None:
                st.markdown("**Validación temporal (ajuste 2021-2022, aplicado a 2023)**")
                st.dataframe(valt.T.rename(columns={0: "valor"}), use_container_width=True)

        st.download_button(
            "⬇️ Descargar asignación GMM (CSV)",
            asig.drop(columns=["etiqueta", "grupo_perfil"]).to_csv(index=False).encode("utf-8"),
            file_name="asignacion_gmm.csv", mime="text/csv",
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Maestría en Inteligencia Analítica de Datos — Universidad de los Andes | "
    "Gerencia de Proyectos de Analytics — 2026"
)