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
# NAVEGACIÓN POR PESTAÑAS
# ============================================================
tab_resumen, tab_ranking, tab_ficha, tab_correlaciones, tab_pca, tab_iqr, tab_clusters, tab_modelo = st.tabs([
    "📊 Resumen",
    "🏆 Ranking",
    "🔍 Ficha Municipal",
    "📈 Correlaciones",
    "🔬 PCA",
    "📦 IQR - Outliers",
    "🎯 Clusters (K-Means)",
    "🧪 Modelo (GMM)",
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
# TAB 2 - RANKING
# ============================================================
with tab_ranking:
    st.subheader("🏆 Ranking de oportunidades de inversión")
    st.info(
        "📌 Municipios ordenados por crecimiento interanual. "
        "🟢 Crecimiento > 5% | 🟡 0–5% | 🔴 Decrecimiento | ⚪ Sin datos.",
        icon="ℹ️",
    )

    df_crec = calcular_crecimiento(df_filtrado)

    if df_crec is not None and len(df_crec) > 0:
        var_rank = st.selectbox("Ordenar por", VARS_NORM, index=1, key="var_rank",
                                format_func=etiqueta)

        df_rank = df_crec.sort_values(var_rank, ascending=False).reset_index(drop=True)
        df_rank.index = df_rank.index + 1

        # Construir tabla con semáforos integrados en el valor
        df_display = pd.DataFrame({"Municipio": df_rank["municipio"]})
        for var in VARS_NORM:
            short = etiqueta(var).replace(" por 1.000 hab.", "")
            df_display[short] = df_rank[var].apply(
                lambda v: f"{semaforo(v)} {v:+.1f}%" if pd.notna(v) else "⚪ N/A"
            )
        df_display.index = df_rank.index

        st.dataframe(df_display, use_container_width=True, hide_index=False)

        # Top 3 y Bottom 3
        col_top, col_bot = st.columns(2)
        with col_top:
            st.markdown("#### 🟢 Top 3 — Mayor crecimiento")
            top3 = df_rank.head(3)[["municipio", var_rank]].reset_index(drop=True)
            top3.columns = ["Municipio", f"{etiqueta(var_rank)} (% anual)"]
            st.dataframe(top3, use_container_width=True, hide_index=True)

        with col_bot:
            st.markdown("#### 🔴 Bottom 3 — Menor crecimiento")
            bot3 = df_rank.tail(3)[["municipio", var_rank]].reset_index(drop=True)
            bot3.columns = ["Municipio", f"{etiqueta(var_rank)} (% anual)"]
            st.dataframe(bot3, use_container_width=True, hide_index=True)
    else:
        st.warning("Se necesitan al menos 2 años seleccionados para calcular crecimiento.")

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
# TAB 5 - PCA
# ============================================================
with tab_pca:
    st.subheader("Análisis de Componentes Principales (PCA)")
    st.info(
        "📌 Reduce las 5 variables normalizadas a 2 componentes principales para "
        "visualizar similitudes entre municipios. Municipios cercanos en el gráfico "
        "tienen perfiles inmobiliarios similares.",
        icon="ℹ️",
    )

    anio_pca = st.selectbox("Año para PCA", anios, index=len(anios) - 1, key="anio_pca")
    df_pca = df[df["anio"] == anio_pca].dropna(subset=VARS_NORM).copy()

    if len(df_pca) >= 3:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_pca[VARS_NORM])

        pca = PCA()
        componentes = pca.fit_transform(X_scaled)

        df_pca["PC1"] = componentes[:, 0]
        df_pca["PC2"] = componentes[:, 1]

        col_scree, col_biplot = st.columns(2)

        with col_scree:
            st.markdown("#### Varianza explicada")
            var_exp = pca.explained_variance_ratio_ * 100
            var_acum = np.cumsum(var_exp)
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(var_exp))],
                y=var_exp, name="Individual",
            ))
            fig_scree.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(var_acum))],
                y=var_acum, name="Acumulada", mode="lines+markers",
            ))
            fig_scree.update_layout(yaxis_title="% Varianza explicada", height=400)
            st.plotly_chart(fig_scree, use_container_width=True)

        with col_biplot:
            st.markdown("#### Biplot (PC1 vs PC2)")
            fig_bi = px.scatter(
                df_pca, x="PC1", y="PC2", text="municipio",
                title=f"Proyección PCA — {anio_pca}",
            )
            loadings = pca.components_[:2, :]
            scale = float(np.abs(componentes[:, :2]).max()) * 0.8
            for i, var in enumerate(VARS_NORM):
                fig_bi.add_annotation(
                    ax=0, ay=0,
                    x=loadings[0, i] * scale, y=loadings[1, i] * scale,
                    showarrow=True, arrowhead=2, arrowsize=1.5,
                    text=etiqueta(var).replace(" por 1.000 hab.", ""),
                )
            fig_bi.update_traces(textposition="top center", marker=dict(size=8))
            fig_bi.update_layout(height=400)
            st.plotly_chart(fig_bi, use_container_width=True)

        with st.expander("Ver cargas por componente"):
            cargas = pd.DataFrame(
                pca.components_.T,
                index=[etiqueta(v) for v in VARS_NORM],
                columns=[f"PC{i+1}" for i in range(len(VARS_NORM))],
            ).round(3)
            st.dataframe(cargas, use_container_width=True)
    else:
        st.warning("No hay suficientes datos para realizar el PCA con los filtros actuales.")

# ============================================================
# TAB 6 - IQR / OUTLIERS
# ============================================================
with tab_iqr:
    st.subheader("Detección de outliers (IQR)")
    st.info(
        "📌 Identifica municipios con valores atípicos (muy altos o bajos) respecto al grupo. "
        "Útil para detectar mercados excepcionales o datos sospechosos.",
        icon="ℹ️",
    )

    anio_iqr = st.selectbox("Año", anios, index=len(anios) - 1, key="anio_iqr")
    var_iqr = st.selectbox("Variable", VARS_NORM, index=0, key="var_iqr",
                           format_func=etiqueta)

    df_iqr = df[df["anio"] == anio_iqr].dropna(subset=[var_iqr]).copy()

    if len(df_iqr) >= 4:
        Q1 = df_iqr[var_iqr].quantile(0.25)
        Q3 = df_iqr[var_iqr].quantile(0.75)
        IQR_val = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR_val
        lim_sup = Q3 + 1.5 * IQR_val

        df_iqr["outlier"] = (df_iqr[var_iqr] < lim_inf) | (df_iqr[var_iqr] > lim_sup)

        col_box, col_tabla = st.columns([1, 1])

        with col_box:
            fig_box = px.box(
                df_iqr, y=var_iqr, points="all", hover_data=["municipio"],
                title=f"Distribución de {etiqueta(var_iqr)} — {anio_iqr}",
                labels={var_iqr: etiqueta(var_iqr)},
            )
            fig_box.update_layout(height=450)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_tabla:
            st.markdown("#### Límites IQR")
            st.write(f"- **Q1:** {Q1:,.2f}")
            st.write(f"- **Q3:** {Q3:,.2f}")
            st.write(f"- **IQR:** {IQR_val:,.2f}")
            st.write(f"- **Límite inferior:** {lim_inf:,.2f}")
            st.write(f"- **Límite superior:** {lim_sup:,.2f}")

            outliers = df_iqr[df_iqr["outlier"]][["municipio", var_iqr]].sort_values(
                var_iqr, ascending=False
            )
            st.markdown(f"#### Outliers detectados ({len(outliers)})")
            if len(outliers) > 0:
                st.dataframe(outliers.rename(columns={var_iqr: etiqueta(var_iqr)}),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No se detectaron outliers con estos filtros.")
    else:
        st.warning("No hay suficientes datos para el análisis IQR.")

# ============================================================
# TAB 7 - CLUSTERS K-MEANS
# ============================================================
with tab_clusters:
    st.subheader("Clustering K-Means")
    st.info(
        "📌 Agrupa municipios con perfiles inmobiliarios similares. "
        "Ajusta k y observa cómo se reagrupan. "
        "El Silhouette Score indica la calidad (más alto = mejor separación).",
        icon="ℹ️",
    )

    anio_km = st.selectbox("Año", anios, index=len(anios) - 1, key="anio_km")
    n_clusters = st.slider("Número de clusters (k)", 2, 6, 3, key="n_clusters")

    df_km = df[df["anio"] == anio_km].dropna(subset=VARS_NORM).copy()

    if len(df_km) >= n_clusters + 1:
        scaler_km = StandardScaler()
        X_km = scaler_km.fit_transform(df_km[VARS_NORM])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_km = kmeans.fit_predict(X_km)
        df_km["cluster"] = labels_km.astype(str)

        # Silhouette Score
        sil_score = silhouette_score(X_km, labels_km)
        st.metric("Silhouette Score", f"{sil_score:.3f}",
                  help="Rango: -1 a 1. Mayor a 0.5 = buena separación. 0.25–0.5 = aceptable.")

        # PCA para visualización 2D
        pca_km = PCA(n_components=2)
        coords = pca_km.fit_transform(X_km)
        df_km["PC1"] = coords[:, 0]
        df_km["PC2"] = coords[:, 1]

        fig_km = px.scatter(
            df_km, x="PC1", y="PC2", color="cluster",
            hover_data=["municipio"] + VARS_NORM, text="municipio",
            title=f"Clusters K-Means (k={n_clusters}) — {anio_km}",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_km.update_traces(textposition="top center", marker=dict(size=10))
        fig_km.update_layout(height=500)
        st.plotly_chart(fig_km, use_container_width=True)

        # Tabla de miembros por cluster
        st.markdown("#### 📋 Municipios por cluster")
        for c in sorted(df_km["cluster"].unique()):
            miembros = df_km[df_km["cluster"] == c]["municipio"].tolist()
            st.markdown(f"**Cluster {c}** ({len(miembros)}): {', '.join(sorted(miembros))}")

        st.markdown("---")

        # Resumen por cluster
        st.markdown("#### Promedios por cluster")
        resumen = df_km.groupby("cluster")[VARS_NORM].mean().round(2)
        resumen_display = resumen.rename(columns={v: etiqueta(v) for v in VARS_NORM})
        st.dataframe(resumen_display, use_container_width=True)

        # Radar por cluster
        st.markdown("#### Perfil de cada cluster")
        rango = resumen.max() - resumen.min()
        rango = rango.replace(0, 1)
        resumen_norm = (resumen - resumen.min()) / rango
        resumen_norm = resumen_norm.fillna(0)

        fig_radar = go.Figure()
        theta_labels = [etiqueta(v).replace(" por 1.000 hab.", "") for v in VARS_NORM]
        theta_labels_closed = theta_labels + [theta_labels[0]]

        for cluster in resumen_norm.index:
            vals = resumen_norm.loc[cluster]
            if isinstance(vals, (int, float, np.floating)):
                r_values = [float(vals)] * len(VARS_NORM) + [float(vals)]
            else:
                r_values = vals.values.tolist() + [vals.values[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=r_values, theta=theta_labels_closed,
                name=f"Cluster {cluster}", fill="toself", opacity=0.5,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Silhouette por k
        with st.expander("🔎 Explorar Silhouette Score por número de clusters"):
            sil_scores = []
            k_range = range(2, min(len(df_km), 7))
            for k in k_range:
                km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                labs = km_temp.fit_predict(X_km)
                sil_scores.append(silhouette_score(X_km, labs))

            fig_sil = px.line(
                x=list(k_range), y=sil_scores,
                markers=True, title="Silhouette Score vs. k",
                labels={"x": "Número de clusters (k)", "y": "Silhouette Score"},
            )
            fig_sil.update_layout(height=300)
            st.plotly_chart(fig_sil, use_container_width=True)
    else:
        st.warning("No hay suficientes datos para el clustering.")

# ============================================================
# TAB 8 - MODELO GMM (placeholder)
# ============================================================
with tab_modelo:
    st.subheader("🧪 Modelo — Mezclas Gaussianas (GMM)")

    st.info(
        "⏳ **Pendiente:** esta sección se habilitará cuando el compañero "
        "entregue los resultados del modelo GMM.\n\n"
        "**Formato esperado:** un CSV con columnas:\n"
        "- `municipio`\n"
        "- `anio`\n"
        "- `cluster_gmm` (cluster asignado)\n"
        "- `prob_cluster_0`, `prob_cluster_1`, ... (probabilidades de pertenencia)"
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Maestría en Inteligencia Analítica de Datos — Universidad de los Andes | "
    "Gerencia de Proyectos de Analytics — 2026"
)