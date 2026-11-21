import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
@st.cache_data
def cargar_datos():
    df = pd.read_excel(datos/Summary municipio-anio 2021-2023.xlsx)
    return df

df = cargar_datos()

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

# ============================================================
# SIDEBAR - FILTROS
# ============================================================
st.sidebar.header("Filtros")

anios = sorted(df["anio"].unique())
anio_sel = st.sidebar.multiselect("Año", anios, default=anios)

municipios = sorted(df["municipio"].unique())
mun_sel = st.sidebar.multiselect("Municipio", municipios, default=municipios)

df_filtrado = df[df["anio"].isin(anio_sel) & df["municipio"].isin(mun_sel)]

st.sidebar.markdown("---")
st.sidebar.metric("Registros filtrados", len(df_filtrado))
st.sidebar.metric("Municipios", df_filtrado["municipio"].nunique())

# ============================================================
# NAVEGACIÓN POR PESTAÑAS
# ============================================================
tab_resumen, tab_pca, tab_iqr, tab_clusters, tab_modelo = st.tabs([
    "📊 Resumen",
    "🔬 PCA",
    "📦 IQR - Outliers",
    "🎯 Clusters (K-Means)",
    "🧪 Modelo (GMM)",  # placeholder para cuando lleguen resultados
])

# ============================================================
# TAB 1 - RESUMEN GENERAL
# ============================================================
with tab_resumen:
    st.subheader("Panorama general")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total registros", f"{df_filtrado['registros'].sum():,.0f}")
    with col2:
        st.metric("Dinámica inmobiliaria", f"{df_filtrado['dinamica'].sum():,.0f}")
    with col3:
        st.metric("Licencias vivienda", f"{df_filtrado['licencias_vivienda'].sum():,.0f}")
    with col4:
        st.metric("Unidades vivienda", f"{df_filtrado['unidades_vivienda'].sum():,.0f}")

    st.markdown("---")

    # Evolución temporal
    st.subheader("Evolución temporal por municipio")
    var_temporal = st.selectbox(
        "Variable",
        VARS_NORM + VARS_ABSOLUTAS,
        index=0,
        key="var_temporal",
    )
    fig_temp = px.line(
        df_filtrado,
        x="anio",
        y=var_temporal,
        color="municipio",
        markers=True,
        title=f"{var_temporal} por año",
    )
    fig_temp.update_layout(height=450)
    st.plotly_chart(fig_temp, use_container_width=True)

    # Comparativo entre municipios
    st.subheader("Comparativo entre municipios")
    var_comparar = st.selectbox(
        "Variable",
        VARS_NORM,
        index=0,
        key="var_comparar",
    )
    anio_comparar = st.selectbox("Año", anios, index=len(anios) - 1, key="anio_comp")
    df_comp = df_filtrado[df_filtrado["anio"] == anio_comparar].sort_values(
        var_comparar, ascending=True
    )
    fig_comp = px.bar(
        df_comp,
        x=var_comparar,
        y="municipio",
        orientation="h",
        title=f"{var_comparar} — {anio_comparar}",
        color=var_comparar,
        color_continuous_scale="Viridis",
    )
    fig_comp.update_layout(height=max(400, len(df_comp) * 25))
    st.plotly_chart(fig_comp, use_container_width=True)

# ============================================================
# TAB 2 - PCA
# ============================================================
with tab_pca:
    st.subheader("Análisis de Componentes Principales (PCA)")

    anio_pca = st.selectbox("Año para PCA", anios, index=len(anios) - 1, key="anio_pca")
    df_pca = df[df["anio"] == anio_pca].dropna(subset=VARS_NORM).copy()

    if len(df_pca) >= 3:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_pca[VARS_NORM])

        pca = PCA()
        componentes = pca.fit_transform(X_scaled)

        df_pca["PC1"] = componentes[:, 0]
        df_pca["PC2"] = componentes[:, 1]

        # Varianza explicada (Scree Plot)
        col_scree, col_biplot = st.columns(2)

        with col_scree:
            st.markdown("#### Varianza explicada")
            var_exp = pca.explained_variance_ratio_ * 100
            var_acum = np.cumsum(var_exp)
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(var_exp))],
                y=var_exp,
                name="Individual",
            ))
            fig_scree.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(var_acum))],
                y=var_acum,
                name="Acumulada",
                mode="lines+markers",
            ))
            fig_scree.update_layout(
                yaxis_title="% Varianza explicada",
                height=400,
            )
            st.plotly_chart(fig_scree, use_container_width=True)

        # Biplot
        with col_biplot:
            st.markdown("#### Biplot (PC1 vs PC2)")
            fig_bi = px.scatter(
                df_pca,
                x="PC1",
                y="PC2",
                text="municipio",
                title=f"Proyección PCA — {anio_pca}",
            )
            # Vectores de carga
            loadings = pca.components_[:2, :]
            scale = float(np.abs(componentes[:, :2]).max()) * 0.8
            for i, var in enumerate(VARS_NORM):
                fig_bi.add_annotation(
                    ax=0, ay=0,
                    x=loadings[0, i] * scale,
                    y=loadings[1, i] * scale,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    text=var.replace("_x1000hab", ""),
                )
            fig_bi.update_traces(textposition="top center", marker=dict(size=8))
            fig_bi.update_layout(height=400)
            st.plotly_chart(fig_bi, use_container_width=True)

        # Tabla de cargas
        with st.expander("Ver cargas por componente"):
            cargas = pd.DataFrame(
                pca.components_.T,
                index=VARS_NORM,
                columns=[f"PC{i+1}" for i in range(len(VARS_NORM))],
            ).round(3)
            st.dataframe(cargas, use_container_width=True)
    else:
        st.warning("No hay suficientes datos para realizar el PCA con los filtros actuales.")

# ============================================================
# TAB 3 - IQR / OUTLIERS
# ============================================================
with tab_iqr:
    st.subheader("Detección de outliers (IQR)")

    anio_iqr = st.selectbox("Año", anios, index=len(anios) - 1, key="anio_iqr")
    var_iqr = st.selectbox("Variable", VARS_NORM, index=0, key="var_iqr")

    df_iqr = df[df["anio"] == anio_iqr].dropna(subset=[var_iqr]).copy()

    if len(df_iqr) >= 4:
        Q1 = df_iqr[var_iqr].quantile(0.25)
        Q3 = df_iqr[var_iqr].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR

        df_iqr["outlier"] = (df_iqr[var_iqr] < lim_inf) | (df_iqr[var_iqr] > lim_sup)

        col_box, col_tabla = st.columns([1, 1])

        with col_box:
            fig_box = px.box(
                df_iqr,
                y=var_iqr,
                points="all",
                hover_data=["municipio"],
                title=f"Distribución de {var_iqr} — {anio_iqr}",
            )
            fig_box.update_layout(height=450)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_tabla:
            st.markdown("#### Límites IQR")
            st.write(f"- **Q1:** {Q1:,.2f}")
            st.write(f"- **Q3:** {Q3:,.2f}")
            st.write(f"- **IQR:** {IQR:,.2f}")
            st.write(f"- **Límite inferior:** {lim_inf:,.2f}")
            st.write(f"- **Límite superior:** {lim_sup:,.2f}")

            outliers = df_iqr[df_iqr["outlier"]][["municipio", var_iqr]].sort_values(
                var_iqr, ascending=False
            )
            st.markdown(f"#### Outliers detectados ({len(outliers)})")
            if len(outliers) > 0:
                st.dataframe(outliers, use_container_width=True, hide_index=True)
            else:
                st.info("No se detectaron outliers con estos filtros.")
    else:
        st.warning("No hay suficientes datos para el análisis IQR.")

# ============================================================
# TAB 4 - CLUSTERS K-MEANS (resultados Guía 4)
# ============================================================
with tab_clusters:
    st.subheader("Clustering K-Means (Guía 4)")

    anio_km = st.selectbox("Año", anios, index=len(anios) - 1, key="anio_km")
    n_clusters = st.slider("Número de clusters", 2, 6, 3, key="n_clusters")

    df_km = df[df["anio"] == anio_km].dropna(subset=VARS_NORM).copy()

    if len(df_km) >= n_clusters:
        scaler_km = StandardScaler()
        X_km = scaler_km.fit_transform(df_km[VARS_NORM])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_km["cluster"] = kmeans.fit_predict(X_km).astype(str)

        # PCA para visualización 2D
        pca_km = PCA(n_components=2)
        coords = pca_km.fit_transform(X_km)
        df_km["PC1"] = coords[:, 0]
        df_km["PC2"] = coords[:, 1]

        fig_km = px.scatter(
            df_km,
            x="PC1",
            y="PC2",
            color="cluster",
            hover_data=["municipio"] + VARS_NORM,
            text="municipio",
            title=f"Clusters K-Means (k={n_clusters}) — {anio_km}",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_km.update_traces(textposition="top center", marker=dict(size=10))
        fig_km.update_layout(height=500)
        st.plotly_chart(fig_km, use_container_width=True)

        # Resumen por cluster
        st.markdown("#### Promedios por cluster")
        resumen = df_km.groupby("cluster")[VARS_NORM].mean().round(2)
        st.dataframe(resumen, use_container_width=True)

        # Radar por cluster
        st.markdown("#### Perfil de cada cluster")
        rango = resumen.max() - resumen.min()
        rango = rango.replace(0, 1)  # evitar división por cero
        resumen_norm = (resumen - resumen.min()) / rango
        resumen_norm = resumen_norm.fillna(0)
        fig_radar = go.Figure()
        theta_labels = [v.replace("_x1000hab", "") for v in VARS_NORM] + [VARS_NORM[0].replace("_x1000hab", "")]
        for cluster in resumen_norm.index:
            vals = resumen_norm.loc[cluster]
            if isinstance(vals, (int, float, np.floating)):
                vals = pd.Series([vals], index=VARS_NORM[:1])
            r_values = vals.values.tolist() + [vals.values[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_labels,
                name=f"Cluster {cluster}",
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.warning("No hay suficientes datos para el clustering.")

# ============================================================
# TAB 5 - MODELO GMM (placeholder)
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

    # --- DESCOMENTAR CUANDO ESTÉN LOS RESULTADOS ---
    # @st.cache_data
    # def cargar_gmm():
    #     return pd.read_csv("data/resultados_gmm.csv")
    #
    # df_gmm = cargar_gmm()
    #
    # fig_gmm = px.scatter(
    #     df_gmm, x="PC1", y="PC2",
    #     color="cluster_gmm",
    #     hover_data=["municipio", "prob_cluster_0", "prob_cluster_1"],
    #     title="Clusters GMM con probabilidades",
    # )
    # st.plotly_chart(fig_gmm, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Maestría en Inteligencia Analítica de Datos — Universidad de los Andes | "
    "Gerencia de Proyectos de Analytics — 2026"
)