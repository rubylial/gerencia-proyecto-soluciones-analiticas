"""Dashboard gerencial del anteproyecto inmobiliario.

Versión integrada a partir del app.py original del equipo: conserva las
ocho secciones y sus controles, con navegación azul y gris uniforme.
Requiere el Excel municipio-año 2021–2023. El GMM se presenta solo cuando
existen resultados reales en dashboard/data/gmm_resultados.csv.
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# CONFIGURACIÓN DEL TABLERO Y ESTILO VISUAL
#-------------------------------------------------------------------
st.set_page_config(
    page_title="Dinámica inmobiliaria territorial | MIAD",
    layout="wide",
    initial_sidebar_state="collapsed")

BASE_DIR = Path(__file__).resolve().parent
RUTAS_DATOS = [
    BASE_DIR.parent / "datos" / "Summary municipio-anio 2021-2023.xlsx",
    BASE_DIR / "data" / "Summary municipio-anio 2021-2023.xlsx"]
RUTA_GMM = BASE_DIR / "data" / "gmm_resultados.csv"
RUTA_GMM_CONFIG = BASE_DIR / "data" / "gmm_config.json"
RUTA_GMM_EVALUACION = BASE_DIR / "data" / "gmm_evaluacion.csv"

AZUL = "#243F5C"
AZUL_MEDIO = "#3C72A8"
FONDO_GENERAL = "#E9EEF4"
FONDO_GRAFICO = "#F5F8FC"
GRIS = "#62758A"
BORDE = "#C8D4E1"
GRILLA = "#DCE4EE"
GRIS_REFERENCIA = "#8997AA"
COLORES = [AZUL_MEDIO, AZUL, "#6D8FAF", "#8D9AB0", "#54708F", "#A7B9CD"]

st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        font-family: "Segoe UI", Arial, sans-serif !important;
    }
    .stApp { background-color: #E9EEF4; color: #243F5C; }
    header[data-testid="stHeader"] { display: none !important; height: 0 !important; }
    [data-testid="stToolbar"], #MainMenu, footer { display: none !important; }
    [data-testid="stMainBlockContainer"], .block-container {
        max-width: none !important;
        width: 100% !important;
        padding: .45rem 1.6rem 1.8rem !important;
    }
    [data-testid="stVerticalBlock"] { gap: .78rem; }
    h1, h2, h3, h4 { font-family: "Segoe UI", Arial, sans-serif !important; color: #243F5C; }
    h3 { font-size: 1.08rem !important; font-weight: 680 !important; margin-top: 0 !important; }
    h4 { font-size: .97rem !important; font-weight: 650 !important; }
    .cabecera-titulo {
        text-align: center; font-family: "Segoe UI", Arial, sans-serif;
        font-size: clamp(1.45rem, 2vw, 2rem); font-weight: 740;
        color: #243F5C; letter-spacing: -.018em; line-height: 1.15;
        margin: .16rem 0 .35rem;
    }
    .cabecera-subtitulo {
        text-align: center; font-size: .83rem; color: #3C72A8;
        font-weight: 610; letter-spacing: .065em; margin: 0 0 .60rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #C8D4E1 !important;
        border-radius: 11px !important;
        box-shadow: 0 2px 9px rgba(36,63,92,.075);
        padding: .1rem .12rem;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #C8D4E1;
        border-radius: 10px; padding: .85rem 1.05rem;
        min-height: 104px; box-shadow: 0 2px 9px rgba(36,63,92,.055);
    }
    [data-testid="stMetricLabel"] { color: #62758A; font-size: .78rem; }
    [data-testid="stMetricValue"] { color: #243F5C; font-size: 1.53rem; font-weight: 720; }
    [data-testid="stMetricDelta"] { font-size: .80rem; }
    /* Pestañas rectangulares de igual anchura, alineadas a todo el tablero. */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: grid !important;
        grid-template-columns: repeat(8, minmax(0, 1fr));
        gap: .42rem !important;
        width: 100% !important;
        border-bottom: none !important;
        padding: .16rem 0 .48rem;
        background-color: transparent;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        min-width: 0 !important;
        min-height: 43px !important;
        padding: .55rem .35rem !important;
        color: #3D526B !important;
        background-color: #FFFFFF !important;
        border: 1px solid #C8D4E1 !important;
        border-radius: 7px !important;
        box-shadow: 0 1px 3px rgba(36,63,92,.04);
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: clamp(.67rem, .77vw, .84rem) !important;
        font-weight: 590 !important;
        line-height: 1.12 !important;
        white-space: normal !important;
        text-align: center !important;
        transition: background-color .12s, border-color .12s;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] p {
        font-size: inherit !important;
        line-height: inherit !important;
        white-space: normal !important;
        text-align: center !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        background-color: #F5F8FC !important;
        border-color: #7193B4 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #243F5C !important;
        border-color: #243F5C !important;
        color: #FFFFFF !important;
        font-weight: 720 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p {
        color: #FFFFFF !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }
    /* Controles de selección múltiple: azul y gris, nunca rojo. */
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background-color: #E0EAF5 !important;
        border: 1px solid #B7CAE0 !important;
        border-radius: 5px !important;
        color: #243F5C !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] *,
    div[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
        color: #243F5C !important;
        fill: #243F5C !important;
    }
    div[data-testid="stMultiSelect"] [data-baseweb="tag"]:hover {
        background-color: #D4E2F0 !important;
    }
    /* Coherencia azul en los controles nativos de Streamlit. */
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #3C72A8 !important;
        border-color: #3C72A8 !important;
        accent-color: #3C72A8 !important;
    }
    div[data-testid="stSlider"] { accent-color: #3C72A8 !important; }
    div[data-testid="stSlider"] [data-testid="stTickBar"] { color: #3C72A8 !important; }
    div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
    div[data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within {
        border-color: #3C72A8 !important;
        box-shadow: 0 0 0 1px #3C72A8 !important;
    }
    @media (max-width: 1120px) {
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: .25rem !important;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            font-size: .68rem !important;
            padding: .45rem .20rem !important;
        }
    }
    @media (max-width: 720px) {
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            display: flex !important;
            overflow-x: auto !important;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            min-width: 122px !important;
        }
    }
    div[data-testid="stSelectbox"] label, div[data-testid="stMultiSelect"] label,
    div[data-testid="stSlider"] label { color: #52677C; font-size: .79rem; font-weight: 620; }
    div[data-baseweb="select"] > div {
        background: #FFFFFF; border-color: #C9D4DF;
        border-radius: 7px; min-height: 37px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #DCE4EE;
        border-radius: 7px; overflow: hidden;
        background-color: #FFFFFF;
    }
    [data-testid="stPlotlyChart"] {
        background-color: #FFFFFF;
        border-radius: 7px;
        overflow: hidden;
    }
    [data-testid="stAlert"] { border-radius: 7px; }
    [data-testid="stDownloadButton"] button {
        background: #FFFFFF; border: 1px solid #C9D4DF;
        color: #243F5C; border-radius: 7px;
    }
    [data-testid="stCaptionContainer"] p { color: #62758A; font-size: .78rem; }
    .seccion-pagina { margin: .22rem 0 .14rem; font-size: 1.10rem; font-weight: 690; color: #243F5C; }
    .pie-equipo { text-align: center; color: #62758A; font-size: .75rem;
        border-top: 1px solid #C8D4E1; padding: 1rem 0 .2rem; }
    section[data-testid="stSidebar"] { display: none; }
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"], .block-container {
            padding: .55rem .65rem 1.1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# VARIABLES E IDENTIFICADORES DEL PROYECTO
#-------------------------------------------------------------------
ETIQUETAS = {
    "registros": "Actos jurídicos registrados",
    "dinamica": "Actos con indicador de dinámica",
    "registros_con_valor": "Registros con valor positivo",
    "licencias_vivienda": "Licencias de vivienda",
    "unidades_vivienda": "Unidades de vivienda autorizadas",
    "unidades_obra_nueva": "Unidades autorizadas de obra nueva",
    "area_vivienda_m2": "Área autorizada de vivienda (m²)",
    "registros_x1000hab": "Actos registrales por 1.000 hab.",
    "dinamica_x1000hab": "Indicador de dinámica por 1.000 hab.",
    "licencias_vivienda_x1000hab": "Licencias de vivienda por 1.000 hab.",
    "unidades_vivienda_x1000hab": "Unidades autorizadas por 1.000 hab.",
    "unidades_obra_nueva_x1000hab": "Obra nueva autorizada por 1.000 hab.",
    "poblacion": "Población",
    "pct_dinamica": "Porcentaje de registros con dinámica",
    "pct_registros_con_valor": "Porcentaje de registros con valor positivo",
    "unidades_novis": "Unidades No VIS",
    "unidades_vip": "Unidades VIP",
    "unidades_vis": "Unidades VIS",
    "unidades_apartamentos": "Unidades de apartamentos",
    "unidades_casas": "Unidades de casas"}

VARS_NORM = [
    "registros_x1000hab", "dinamica_x1000hab", "licencias_vivienda_x1000hab",
    "unidades_vivienda_x1000hab", "unidades_obra_nueva_x1000hab"]
VARS_ABSOLUTAS = [
    "registros", "dinamica", "registros_con_valor", "licencias_vivienda",
    "unidades_vivienda", "unidades_obra_nueva", "area_vivienda_m2"]
VALLE = {
    "05001", "05079", "05088", "05129", "05212", "05266",
    "05308", "05360", "05380", "05631"}
COORDS = {
    "05001": (6.2442, -75.5812), "05079": (6.4385, -75.3314),
    "05088": (6.3373, -75.5579), "05129": (6.0916, -75.6368),
    "05212": (6.3464, -75.5085), "05266": (6.1690, -75.5870),
    "05308": (6.3774, -75.4478), "05318": (6.2815, -75.4439),
    "05360": (6.1726, -75.6113), "05376": (6.0326, -75.4303),
    "05380": (6.1571, -75.6431), "05440": (6.1737, -75.3358),
    "05607": (6.0578, -75.5014), "05615": (6.1551, -75.3737),
    "05631": (6.1515, -75.6166), "11001": (4.7110, -74.0721)}


def etiqueta(variable):
    return ETIQUETAS.get(variable, variable)


def numero(valor, decimales=0):
    if pd.isna(valor):
        return "—"
    return f"{valor:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# CARGA DE DATOS Y CONTROLES DE INTEGRIDAD
#-------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_datos(ruta):
    base = pd.read_excel(ruta, dtype={"divipola": str})
    indispensables = {"divipola", "municipio", "anio", "poblacion", "registros",
        "dinamica", "registros_con_valor", "licencias_vivienda", "unidades_vivienda",
        *VARS_NORM, *VARS_ABSOLUTAS}
    faltantes = sorted(indispensables - set(base.columns))
    if faltantes:
        raise ValueError(f"El resumen no contiene estas columnas: {', '.join(faltantes)}")
    base["divipola"] = base["divipola"].astype(str).str.zfill(5)
    base["anio"] = pd.to_numeric(base["anio"], errors="raise").astype(int)
    if base.duplicated(["divipola", "anio"]).any():
        raise ValueError("La base integrada contiene combinaciones municipio-año repetidas.")
    base["ambito"] = np.where(base["divipola"].isin(VALLE), "Valle de Aburrá", "Referencia")
    base["lat"] = base["divipola"].map({c: v[0] for c, v in COORDS.items()})
    base["lon"] = base["divipola"].map({c: v[1] for c, v in COORDS.items()})
    return base


@st.cache_data(show_spinner=False)
def cargar_gmm(ruta, fecha_modificacion=None):
    if not Path(ruta).is_file():
        return None
    gmm = pd.read_csv(ruta, dtype={"municipio": str})
    requeridas = {"municipio", "cluster_gmm"}
    if not requeridas.issubset(gmm.columns):
        raise ValueError("El archivo GMM debe incluir municipio y cluster_gmm.")
    gmm["municipio"] = gmm["municipio"].str.strip()
    if "anio" in gmm.columns:
        gmm["anio"] = pd.to_numeric(gmm["anio"], errors="raise").astype(int)
    claves = ["municipio", "anio"] if "anio" in gmm.columns else ["municipio"]
    if gmm.duplicated(claves).any():
        raise ValueError("Hay municipios-año repetidos en los resultados GMM.")
    probabilidades = sorted([col for col in gmm.columns if re.fullmatch(r"prob_cluster_\d+", col)],
        key=lambda col: int(col.rsplit("_", 1)[1]))
    for col in probabilidades:
        gmm[col] = pd.to_numeric(gmm[col], errors="raise")
    if probabilidades:
        matriz = gmm[probabilidades].to_numpy(dtype=float)
        if not np.isfinite(matriz).all() or np.any(matriz < 0) or np.any(matriz > 1):
            raise ValueError("Las probabilidades GMM deben estar entre 0 y 1.")
        if not np.allclose(matriz.sum(axis=1), 1, atol=1e-3):
            raise ValueError("Las probabilidades GMM deben sumar 1 por observación.")
    gmm["cluster_gmm"] = gmm["cluster_gmm"].astype(str)
    extras = [col for col in ("PC1", "PC2") if col in gmm.columns]
    gmm = gmm[claves + ["cluster_gmm"] + probabilidades + extras].copy()
    return gmm, claves, probabilidades


@st.cache_data(show_spinner=False)
def cargar_config_gmm(ruta, fecha_modificacion=None):
    if not Path(ruta).is_file():
        return None
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


@st.cache_data(show_spinner=False)
def cargar_evaluacion_gmm(ruta, fecha_modificacion=None):
    if not Path(ruta).is_file():
        return None
    return pd.read_csv(ruta)


ruta = next((r for r in RUTAS_DATOS if r.is_file()), None)
if ruta is None:
    st.error("No se encontró 'Summary municipio-anio 2021-2023.xlsx' en datos/ ni dashboard/data/.")
    st.stop()
try:
    df = cargar_datos(str(ruta))
except (ValueError, OSError) as error:
    st.error(f"No fue posible cargar la base: {error}")
    st.stop()
if df.empty:
    st.error("El resumen integrado está vacío.")
    st.stop()


# FORMATO COMÚN PARA GRÁFICOS Y TABLAS
#-------------------------------------------------------------------
def estilo_grafico(fig, alto=360, horizontal=False):
    fig.update_layout(
        height=alto,
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor=FONDO_GRAFICO,
        font=dict(family="Segoe UI, Arial, sans-serif", size=11, color=AZUL),
        margin=dict(l=10, r=16, t=18, b=32),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=AZUL),
        legend=dict(title=None, orientation="h", y=-.22))
    fig.update_xaxes(showgrid=not horizontal, gridcolor=GRILLA,
        zeroline=False, linecolor=GRILLA)
    fig.update_yaxes(showgrid=horizontal, gridcolor=GRILLA,
        zeroline=False, linecolor=GRILLA)
    return fig


def mostrar(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def tabla_datos(dataframe, alto=None):
    opciones = {"width": "stretch", "hide_index": True}
    if alto is not None:
        opciones["height"] = alto
    st.dataframe(dataframe, **opciones)


def indicador(valor, variable, anterior=None):
    cambio = None
    if anterior is not None and anterior > 0:
        cambio = f"{((valor / anterior) - 1) * 100:+.1f}%"
    st.metric(etiqueta(variable), numero(valor), delta=cambio)


def titulo_panel(texto):
    st.markdown(f"#### {texto}")


def exportar_csv(dataframe, nombre):
    st.download_button("Descargar CSV", dataframe.to_csv(index=False).encode("utf-8-sig"),
        file_name=nombre, mime="text/csv")


def calcular_crecimiento(base):
    """Tasa media anual entre el primer y el último año seleccionado."""
    periodos = sorted(base["anio"].unique())
    if len(periodos) < 2:
        return None
    inicial, final = periodos[0], periodos[-1]
    diferencia_anios = final - inicial
    origen = base[base["anio"] == inicial].set_index("municipio")
    destino = base[base["anio"] == final].set_index("municipio")
    comunes = origen.index.intersection(destino.index)
    resultados = []
    for municipio in comunes:
        fila = {"municipio": municipio}
        for variable in VARS_NORM:
            inicio = origen.loc[municipio, variable]
            cierre = destino.loc[municipio, variable]
            fila[variable] = ((cierre / inicio) ** (1 / diferencia_anios) - 1) * 100 if inicio > 0 and cierre >= 0 else np.nan
        resultados.append(fila)
    return pd.DataFrame(resultados)


def clasificar_variacion(valor):
    if pd.isna(valor):
        return "Sin dato"
    if valor > 5:
        return "Aumento > 5 %"
    if valor >= 0:
        return "Entre 0 y 5 %"
    return "Disminución"


def calcular_silueta(datos, etiquetas):
    if 1 < len(np.unique(etiquetas)) < len(etiquetas):
        return float(silhouette_score(datos, etiquetas))
    return np.nan


def centro_mapa(base):
    validos = base.dropna(subset=["lat", "lon"])
    if validos.empty:
        return dict(lat=6.25, lon=-75.57), 8.5
    if validos["lon"].max() - validos["lon"].min() > 1:
        return dict(lat=5.25, lon=-74.9), 5.7
    if len(validos) == 1:
        return dict(lat=float(validos.iloc[0]["lat"].item()),
            lon=float(validos.iloc[0]["lon"].item())), 10
    return dict(lat=float(validos["lat"].mean()), lon=float(validos["lon"].mean())), 8.7


# ENCABEZADO, FILTROS Y DESCARGA
#-------------------------------------------------------------------
st.markdown("""
<div class="cabecera-titulo">DINÁMICA INMOBILIARIA TERRITORIAL</div>
<div class="cabecera-subtitulo">VALLE DE ABURRÁ | ANÁLISIS COMPARATIVO 2021–2023 | IGAC Y DANE</div>
""", unsafe_allow_html=True)

periodos = sorted(int(a) for a in df["anio"].unique())
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.12, 1.15, 2.40, 2.15], gap="medium")
    with c1:
        ambito = st.selectbox("Cobertura", ["Valle de Aburrá", "Referencia", "Todos los territorios"], key="ambito")
    with c2:
        anios_sel = st.multiselect("Años", periodos, default=periodos, key="anios_global")
    df_alcance = df.copy()
    if ambito != "Todos los territorios":
        df_alcance = df_alcance[df_alcance["ambito"] == ambito]
    opciones_municipios = sorted(df_alcance["municipio"].unique())
    with c3:
        mun_sel = st.multiselect("Municipios", opciones_municipios,
            default=[], placeholder="Todos los municipios de la cobertura", key="municipios_global")
    with c4:
        variable_general = st.selectbox("Indicador de referencia", VARS_NORM,
            index=3, format_func=etiqueta, key="indicador_general")
    # Selección vacía de municipios equivale a seleccionar todos.
    df_filtrado = df_alcance[df_alcance["anio"].isin(anios_sel)].copy()
    if mun_sel:
        df_filtrado = df_filtrado[df_filtrado["municipio"].isin(mun_sel)].copy()

if df_filtrado.empty:
    st.warning("La combinación de filtros no contiene observaciones. Selecciona al menos un año.")
    st.stop()

anios_disponibles = sorted(df_filtrado["anio"].unique().tolist())
municipios_disponibles = sorted(df_filtrado["municipio"].unique().tolist())
ultimo_anio = anios_disponibles[-1]

# La exportación se mantiene, pero fuera de la zona de filtros para ahorrar espacio.
with st.popover("Exportar datos del filtro"):
    st.caption("Descarga los registros municipio-año incluidos en los filtros actuales.")
    exportar_csv(df_filtrado.drop(columns=["lat", "lon"]), "municipios_filtrados.csv")


# NAVEGACIÓN SUPERIOR ENTRE LOS OCHO TABLEROS
#-------------------------------------------------------------------
tab_panorama, tab_comparacion, tab_ficha, tab_corr, tab_pca, tab_iqr, tab_kmeans, tab_gmm = st.tabs([
    "Resumen general", "Comparación territorial", "Perfil municipal", "Relación de variables",
    "Componentes (PCA)", "Valores atípicos", "Grupos (K-Means)", "Modelo GMM"])


# TABLERO 1. PANORAMA Y EVOLUCIÓN MUNICIPAL
#-------------------------------------------------------------------
with tab_panorama:
    st.markdown('<div class="seccion-pagina">Resumen de actividad territorial</div>', unsafe_allow_html=True)
    anio_previo = ultimo_anio - 1
    datos_ult = df_filtrado[df_filtrado["anio"] == ultimo_anio]
    datos_prev = df_filtrado[df_filtrado["anio"] == anio_previo]
    indicadores_kpi = ["registros", "dinamica", "licencias_vivienda", "unidades_vivienda"]
    kpis = st.columns(4, gap="medium")
    for i, variable in enumerate(indicadores_kpi):
        total = float(datos_ult[variable].sum())
        anterior = float(datos_prev[variable].sum()) if not datos_prev.empty else None
        with kpis[i]:
            indicador(total, variable, anterior)

    mapa_col, comp_col = st.columns([1.05, 1.10], gap="medium")
    with mapa_col:
        with st.container(border=True):
            titulo_panel("Localización municipal")
            filtros_mapa = st.columns([1.7, 1], gap="small")
            with filtros_mapa[0]:
                variable_mapa = st.selectbox("Indicador del mapa", VARS_NORM,
                    index=VARS_NORM.index(variable_general), format_func=etiqueta, key="var_mapa")
            with filtros_mapa[1]:
                anio_mapa = st.selectbox("Año", anios_disponibles,
                    index=len(anios_disponibles) - 1, key="anio_mapa")
            mapa = df_filtrado[df_filtrado["anio"] == anio_mapa].copy()
            mapa = mapa.dropna(subset=["lat", "lon"])
            if not mapa.empty:
                centro, zoom = centro_mapa(mapa)
                figura_mapa = px.scatter_map(
                    mapa, lat="lat", lon="lon", size=variable_mapa, color="ambito",
                    hover_name="municipio", custom_data=[variable_mapa, "poblacion"],
                    color_discrete_map={"Valle de Aburrá": AZUL_MEDIO, "Referencia": GRIS_REFERENCIA},
                    center=centro, zoom=zoom, map_style="carto-positron", size_max=23)
                figura_mapa.update_traces(hovertemplate="<b>%{hovertext}</b><br>" +
                    etiqueta(variable_mapa) + ": %{customdata[0]:,.2f}<br>Población: %{customdata[1]:,.0f}<extra></extra>")
                figura_mapa.update_layout(height=430, paper_bgcolor="#FFFFFF",
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(title=None, orientation="h", x=0, y=1.02),
                    font=dict(family="Segoe UI, Arial", color=AZUL))
                mostrar(figura_mapa)
                st.caption("Puntos urbanos aproximados; no representan límites municipales oficiales.")
            else:
                st.info("No hay coordenadas disponibles para la selección.")

    with comp_col:
        with st.container(border=True):
            titulo_panel("Comparación por municipio")
            filtros_comp = st.columns([1.7, 1], gap="small")
            with filtros_comp[0]:
                var_comparar = st.selectbox("Indicador comparado", VARS_NORM,
                    index=VARS_NORM.index(variable_general), format_func=etiqueta, key="var_comparar")
            with filtros_comp[1]:
                anio_comparar = st.selectbox("Año", anios_disponibles,
                    index=len(anios_disponibles) - 1, key="anio_comp")
            comparacion = df_filtrado[df_filtrado["anio"] == anio_comparar].sort_values(var_comparar)
            if not comparacion.empty:
                figura_comp = px.bar(comparacion, x=var_comparar, y="municipio",
                    orientation="h", color="ambito",
                    color_discrete_map={"Valle de Aburrá": AZUL_MEDIO, "Referencia": GRIS_REFERENCIA},
                    labels={var_comparar: etiqueta(var_comparar), "municipio": ""},
                    text_auto=".2f")
                figura_comp.update_traces(textposition="outside", cliponaxis=False)
                figura_comp.update_layout(showlegend=False)
                mostrar(estilo_grafico(figura_comp, alto=442))

    with st.container(border=True):
        titulo_panel("Evolución temporal")
        t1, t2 = st.columns([1.15, 2.3], gap="medium")
        with t1:
            var_temporal = st.selectbox("Indicador", VARS_NORM + VARS_ABSOLUTAS,
                format_func=etiqueta, key="var_temporal")
        with t2:
            mun_temporales = st.multiselect("Municipios para comparar", municipios_disponibles,
                default=[], placeholder="Todos los municipios de la cobertura", key="mun_temporal")
        serie = df_filtrado.copy()
        if mun_temporales:
            serie = serie[serie["municipio"].isin(mun_temporales)]
        figura_temp = px.line(serie, x="anio", y=var_temporal, color="municipio",
            markers=True, labels={"anio": "Año", var_temporal: etiqueta(var_temporal),
                "municipio": "Municipio"}, color_discrete_sequence=COLORES)
        figura_temp.update_xaxes(tickmode="array", tickvals=anios_disponibles)
        mostrar(estilo_grafico(figura_temp, alto=370))


# TABLERO 2. CRECIMIENTO Y ORDENAMIENTO COMPARATIVO
#-------------------------------------------------------------------
with tab_comparacion:
    st.markdown('<div class="seccion-pagina">Comparación entre municipios</div>', unsafe_allow_html=True)
    crecimiento = calcular_crecimiento(df_filtrado)
    if crecimiento is None or crecimiento.empty:
        st.info("Selecciona al menos dos años para comparar el crecimiento de las tasas.")
    else:
        with st.container(border=True):
            titulo_panel("Variación media anual por municipio")
            var_rank = st.selectbox("Ordenar por", VARS_NORM, index=1,
                format_func=etiqueta, key="var_rank")
            ranking = crecimiento.sort_values(var_rank, ascending=False).reset_index(drop=True)
            ranking.index = ranking.index + 1
            tabla = ranking.rename(columns={v: etiqueta(v) for v in VARS_NORM})
            tabla.insert(0, "Posición", range(1, len(tabla) + 1))
            tabla["Tendencia"] = ranking[var_rank].map(clasificar_variacion).values
            formato = {etiqueta(v): "{:+.1f} %" for v in VARS_NORM}
            tabla_datos(tabla.style.format(formato, na_rep="—"), alto=min(580, 80 + 36 * len(tabla)))
            st.caption("Variación media anual entre el primer y último año seleccionados. No equivale a un ranking de rentabilidad.")
        top_col, bajo_col = st.columns(2, gap="medium")
        with top_col:
            with st.container(border=True):
                titulo_panel("Tres mayores crecimientos")
                t = ranking.head(3)[["municipio", var_rank]].rename(columns={
                    "municipio": "Municipio", var_rank: "Variación anual (%)"})
                tabla_datos(t.style.format({"Variación anual (%)": "{:+.1f} %"}))
        with bajo_col:
            with st.container(border=True):
                titulo_panel("Tres menores crecimientos")
                t = ranking.tail(3)[["municipio", var_rank]].sort_values(var_rank).rename(columns={
                    "municipio": "Municipio", var_rank: "Variación anual (%)"})
                tabla_datos(t.style.format({"Variación anual (%)": "{:+.1f} %"}))


# TABLERO 3. FICHA MUNICIPAL, PERCENTILES Y RADAR
#-------------------------------------------------------------------
with tab_ficha:
    st.markdown('<div class="seccion-pagina">Perfil del municipio</div>', unsafe_allow_html=True)
    selector_ficha = st.selectbox("Municipio de análisis", municipios_disponibles, key="mun_ficha")
    df_mun = df_filtrado[df_filtrado["municipio"] == selector_ficha].sort_values("anio")
    if df_mun.empty:
        st.info("No hay datos para el municipio seleccionado.")
    else:
        ultimo = df_mun.iloc[-1]
        previo = df_mun[df_mun["anio"] == int(ultimo["anio"]) - 1]
        c_ficha = st.columns(5, gap="small")
        for i, variable in enumerate(VARS_NORM):
            valor = float(ultimo[variable])
            ant = float(previo.iloc[0][variable]) if not previo.empty else None
            variacion = f"{((valor / ant) - 1) * 100:+.1f}%" if ant is not None and ant > 0 else None
            with c_ficha[i]:
                st.metric(etiqueta(variable).replace(" por 1.000 hab.", " / 1.000 hab."),
                    numero(valor, 2), delta=variacion)

        col_evol, col_percentil = st.columns(2, gap="medium")
        with col_evol:
            with st.container(border=True):
                titulo_panel("Evolución del indicador")
                var_ficha = st.selectbox("Indicador", VARS_NORM, index=1,
                    format_func=etiqueta, key="var_ficha_ev")
                figura_ev = px.bar(df_mun, x="anio", y=var_ficha,
                    labels={"anio": "Año", var_ficha: etiqueta(var_ficha)},
                    color_discrete_sequence=[AZUL_MEDIO], text_auto=".2f")
                figura_ev.update_xaxes(tickmode="array", tickvals=anios_disponibles)
                mostrar(estilo_grafico(figura_ev, alto=345))

        with col_percentil:
            with st.container(border=True):
                titulo_panel("Posición relativa en el territorio")
                todos = df_filtrado[df_filtrado["anio"] == int(ultimo["anio"])]
                percentiles = []
                for variable in VARS_NORM:
                    p = (todos[variable] <= ultimo[variable]).sum() / len(todos) * 100
                    percentiles.append({"Variable": etiqueta(variable), "Percentil": p})
                percentiles = pd.DataFrame(percentiles)
                figura_perc = px.bar(percentiles.sort_values("Percentil"),
                    x="Percentil", y="Variable", orientation="h", range_x=[0, 105],
                    color_discrete_sequence=[AZUL_MEDIO], text_auto=".0f")
                mostrar(estilo_grafico(figura_perc, alto=395))
                st.caption(f"Posición calculada entre {len(todos)} municipios del ámbito para {int(ultimo['anio'])}.")

        with st.container(border=True):
            titulo_panel("Perfil municipal frente al promedio")
            todos = df_filtrado[df_filtrado["anio"] == int(ultimo["anio"])]
            promedio = todos[VARS_NORM].mean()
            minimo = todos[VARS_NORM].min()
            rango = (todos[VARS_NORM].max() - minimo).replace(0, 1)
            prom_norm = ((promedio - minimo) / rango).clip(0, 1).to_numpy()
            mun_norm = ((ultimo[VARS_NORM].astype(float) - minimo) / rango).clip(0, 1).to_numpy()
            theta = [etiqueta(v).replace(" por 1.000 hab.", "") for v in VARS_NORM]
            figura_radar = go.Figure()
            figura_radar.add_trace(go.Scatterpolar(r=list(prom_norm) + [prom_norm[0]],
                theta=theta + [theta[0]], name="Promedio del ámbito", fill="toself",
                line_color=AZUL, opacity=.65))
            figura_radar.add_trace(go.Scatterpolar(r=list(mun_norm) + [mun_norm[0]],
                theta=theta + [theta[0]], name=selector_ficha, fill="toself",
                line_color=AZUL_MEDIO, opacity=.65))
            figura_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True)
            mostrar(estilo_grafico(figura_radar, alto=400))
            st.caption("Escala 0–1 calculada por indicador entre los municipios seleccionados. No es una puntuación de inversión.")


# TABLERO 4. CORRELACIONES Y RELACIÓN ENTRE VARIABLES
#-------------------------------------------------------------------
with tab_corr:
    st.markdown('<div class="seccion-pagina">Relación entre indicadores</div>', unsafe_allow_html=True)
    with st.container(border=True):
        f1, f2 = st.columns([1, 3], gap="medium")
        with f1:
            anio_corr = st.selectbox("Año", anios_disponibles,
                index=len(anios_disponibles) - 1, key="anio_corr")
        with f2:
            vars_corr = st.multiselect("Variables a incluir", VARS_NORM + VARS_ABSOLUTAS,
                default=VARS_NORM, format_func=etiqueta, key="vars_corr")
        if len(vars_corr) >= 2:
            base_corr = df_filtrado[df_filtrado["anio"] == anio_corr].dropna(subset=vars_corr)
            if len(base_corr) >= 3:
                matriz = base_corr[vars_corr].corr().round(2)
                nombres = [etiqueta(v) for v in vars_corr]
                fig_corr = px.imshow(matriz, x=nombres, y=nombres, zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto")
                mostrar(estilo_grafico(fig_corr, alto=max(420, len(vars_corr) * 39 + 130)))
                st.caption("Coeficiente de correlación de Pearson. Asociación estadística no implica causalidad.")
            else:
                st.info("Se requieren al menos tres observaciones completas.")
        else:
            st.info("Selecciona al menos dos variables para la matriz.")

    with st.container(border=True):
        titulo_panel("Comparación entre dos indicadores")
        if len(vars_corr) >= 2:
            fx, fy = st.columns(2, gap="medium")
            with fx:
                var_x = st.selectbox("Eje horizontal", vars_corr, index=0,
                    format_func=etiqueta, key="scatter_x")
            with fy:
                var_y = st.selectbox("Eje vertical", vars_corr, index=min(1, len(vars_corr) - 1),
                    format_func=etiqueta, key="scatter_y")
            datos_corr = df_filtrado[df_filtrado["anio"] == anio_corr].dropna(subset=[var_x, var_y])
            if len(datos_corr) >= 3 and var_x != var_y:
                try:
                    figura_disp = px.scatter(datos_corr, x=var_x, y=var_y,
                        hover_name="municipio", trendline="ols",
                        labels={var_x: etiqueta(var_x), var_y: etiqueta(var_y)},
                        color_discrete_sequence=[AZUL_MEDIO])
                except (ImportError, ValueError):
                    figura_disp = px.scatter(datos_corr, x=var_x, y=var_y,
                        hover_name="municipio",
                        labels={var_x: etiqueta(var_x), var_y: etiqueta(var_y)},
                        color_discrete_sequence=[AZUL_MEDIO])
                figura_disp.update_traces(marker=dict(size=10))
                mostrar(estilo_grafico(figura_disp, alto=380))
            else:
                st.info("Elige dos variables distintas y al menos tres observaciones.")
        else:
            st.info("Selecciona al menos dos variables en el filtro superior.")


# TABLERO 5. PCA, VARIANZA EXPLICADA Y CARGAS
#-------------------------------------------------------------------
with tab_pca:
    st.markdown('<div class="seccion-pagina">Componentes principales (PCA)</div>', unsafe_allow_html=True)
    with st.container(border=True):
        anio_pca = st.selectbox("Año", anios_disponibles,
            index=len(anios_disponibles) - 1, key="anio_pca")
        base_pca = df_filtrado[df_filtrado["anio"] == anio_pca].dropna(subset=VARS_NORM).copy()
        if len(base_pca) >= 3:
            estandarizador = StandardScaler()
            x_pca = estandarizador.fit_transform(base_pca[VARS_NORM])
            modelo_pca = PCA()
            componentes = modelo_pca.fit_transform(x_pca)
            base_pca["PC1"] = componentes[:, 0]
            base_pca["PC2"] = componentes[:, 1]
            var_ind = modelo_pca.explained_variance_ratio_ * 100
            var_acum = np.cumsum(var_ind)
            pc1, pc2 = st.columns(2, gap="medium")
            with pc1:
                titulo_panel("Varianza explicada")
                x_componentes = [f"PC{i+1}" for i in range(len(var_ind))]
                fig_scree = go.Figure()
                fig_scree.add_trace(go.Bar(x=x_componentes, y=var_ind,
                    name="Individual", marker_color=AZUL_MEDIO))
                fig_scree.add_trace(go.Scatter(x=x_componentes, y=var_acum,
                    name="Acumulada", mode="lines+markers", line_color=AZUL))
                fig_scree.update_yaxes(title_text="Varianza explicada (%)", range=[0, 105])
                mostrar(estilo_grafico(fig_scree, alto=390))
            with pc2:
                titulo_panel("Representación de los componentes PC1 y PC2")
                fig_biplot = px.scatter(base_pca, x="PC1", y="PC2", text="municipio",
                    hover_name="municipio", color_discrete_sequence=[AZUL_MEDIO])
                cargas = modelo_pca.components_[:2, :]
                factor = float(np.abs(componentes[:, :2]).max()) * .75
                for i, variable in enumerate(VARS_NORM):
                    fig_biplot.add_annotation(ax=0, ay=0,
                        x=float(cargas[0, i] * factor), y=float(cargas[1, i] * factor),
                        showarrow=True, arrowhead=2, text=etiqueta(variable).replace(" por 1.000 hab.", ""),
                        font=dict(size=9, color=AZUL))
                fig_biplot.update_traces(textposition="top center", marker=dict(size=9))
                mostrar(estilo_grafico(fig_biplot, alto=390))
            with st.expander("Ver tabla de cargas por componente"):
                pesos = pd.DataFrame(modelo_pca.components_.T,
                    index=[etiqueta(v) for v in VARS_NORM],
                    columns=[f"PC{i+1}" for i in range(modelo_pca.components_.shape[0])]).round(3)
                tabla_datos(pesos.reset_index().rename(columns={"index": "Indicador"}))
            st.caption("PCA exploratorio de cinco tasas para el año seleccionado; no reproduce necesariamente el PCA de siete variables del informe.")
        else:
            st.info("Se requieren al menos tres municipios con información completa.")


# TABLERO 6. IQR MUNICIPAL Y REGISTROS ATÍPICOS
#-------------------------------------------------------------------
with tab_iqr:
    st.markdown('<div class="seccion-pagina">Valores atípicos en indicadores municipales (IQR)</div>', unsafe_allow_html=True)
    with st.container(border=True):
        filt_iqr = st.columns([1, 2], gap="medium")
        with filt_iqr[0]:
            anio_iqr = st.selectbox("Año", anios_disponibles,
                index=len(anios_disponibles) - 1, key="anio_iqr")
        with filt_iqr[1]:
            var_iqr = st.selectbox("Indicador municipal", VARS_NORM,
                format_func=etiqueta, key="var_iqr")
        base_iqr = df_filtrado[df_filtrado["anio"] == anio_iqr].dropna(subset=[var_iqr]).copy()
        if len(base_iqr) >= 4:
            q1 = float(base_iqr[var_iqr].quantile(.25))
            q3 = float(base_iqr[var_iqr].quantile(.75))
            ri = q3 - q1
            lim_inf = q1 - 1.5 * ri
            lim_sup = q3 + 1.5 * ri
            base_iqr["atipico"] = (base_iqr[var_iqr] < lim_inf) | (base_iqr[var_iqr] > lim_sup)
            zona_box, zona_lim = st.columns([1.6, 1], gap="medium")
            with zona_box:
                titulo_panel("Distribución de los municipios")
                fig_box = px.box(base_iqr, y=var_iqr, points="all",
                    hover_data=["municipio"], labels={var_iqr: etiqueta(var_iqr)},
                    color_discrete_sequence=[AZUL_MEDIO])
                mostrar(estilo_grafico(fig_box, alto=400))
            with zona_lim:
                titulo_panel("Cuartiles y límites IQR")
                limites = pd.DataFrame({"Estadístico": ["Primer cuartil (Q1)", "Tercer cuartil (Q3)",
                    "Rango intercuartílico", "Límite inferior", "Límite superior"],
                    "Valor": [q1, q3, ri, lim_inf, lim_sup]})
                tabla_datos(limites.style.format({"Valor": "{:,.2f}"}))
                atipicos = base_iqr[base_iqr["atipico"]][["municipio", var_iqr]]
                titulo_panel(f"Municipios señalados ({len(atipicos)})")
                if atipicos.empty:
                    st.caption("No se identificaron valores atípicos con esta selección.")
                else:
                    tabla_datos(atipicos.rename(columns={"municipio": "Municipio",
                        var_iqr: etiqueta(var_iqr)}))
            st.caption("IQR sobre indicadores municipio-año. Es distinto del IQR monetario de compraventas desarrollado en la Guía 4.")
        else:
            st.info("Se requieren al menos cuatro municipios para aplicar esta comparación.")


# TABLERO 7. K-MEANS, PERFILES Y EVALUACIÓN DE K
#-------------------------------------------------------------------
with tab_kmeans:
    st.markdown('<div class="seccion-pagina">Grupos de municipios (K-Means)</div>', unsafe_allow_html=True)
    with st.container(border=True):
        controles_km = st.columns([1, 2], gap="medium")
        with controles_km[0]:
            anio_km = st.selectbox("Año", anios_disponibles,
                index=len(anios_disponibles) - 1, key="anio_km")
        base_km = df_filtrado[df_filtrado["anio"] == anio_km].dropna(subset=VARS_NORM).copy()
        max_k = min(6, len(base_km) - 1)
        with controles_km[1]:
            if max_k >= 2:
                numero_clusters = st.slider("Número de grupos (k)", 2, max_k,
                    value=min(3, max_k), key="n_clusters")
            else:
                numero_clusters = None
        if numero_clusters is not None:
            esc = StandardScaler()
            x_km = esc.fit_transform(base_km[VARS_NORM])
            km = KMeans(n_clusters=numero_clusters, random_state=42, n_init=10)
            etiquetas_km = km.fit_predict(x_km)
            base_km["grupo"] = etiquetas_km.astype(str)
            sil = calcular_silueta(x_km, etiquetas_km)
            st.metric("Coeficiente de silueta", numero(sil, 3))
            pc = PCA(n_components=2).fit_transform(x_km)
            base_km["PC1"] = pc[:, 0]
            base_km["PC2"] = pc[:, 1]
            grupo_col, miembros_col = st.columns([1.25, 1], gap="medium")
            with grupo_col:
                titulo_panel("Distribución de municipios por grupo")
                figura_km = px.scatter(base_km, x="PC1", y="PC2", color="grupo",
                    text="municipio", hover_data=VARS_NORM,
                    color_discrete_sequence=COLORES)
                figura_km.update_traces(textposition="top center", marker=dict(size=10))
                mostrar(estilo_grafico(figura_km, alto=440))
            with miembros_col:
                titulo_panel("Municipios por grupo")
                for grupo in sorted(base_km["grupo"].unique()):
                    lista = ", ".join(sorted(base_km[base_km["grupo"] == grupo]["municipio"].tolist()))
                    st.markdown(f"**Grupo {grupo}** | {lista}")
                titulo_panel("Promedios de los grupos")
                promedio_km = base_km.groupby("grupo")[VARS_NORM].mean().round(2)
                promedio_vis = promedio_km.rename(columns={v: etiqueta(v) for v in VARS_NORM})
                tabla_datos(promedio_vis.reset_index().rename(columns={"grupo": "Grupo"}))

            with st.container(border=True):
                titulo_panel("Perfil relativo de cada grupo")
                rangos = (promedio_km.max() - promedio_km.min()).replace(0, 1)
                perfil = ((promedio_km - promedio_km.min()) / rangos).fillna(0)
                theta = [etiqueta(v).replace(" por 1.000 hab.", "") for v in VARS_NORM]
                figura_radar = go.Figure()
                for j, grupo in enumerate(perfil.index):
                    vals = perfil.loc[grupo].to_numpy()
                    figura_radar.add_trace(go.Scatterpolar(r=list(vals) + [vals[0]],
                        theta=theta + [theta[0]], name=f"Grupo {grupo}", fill="toself",
                        line_color=COLORES[j % len(COLORES)], opacity=.56))
                figura_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
                mostrar(estilo_grafico(figura_radar, alto=365))
                st.caption("Perfiles normalizados entre grupos para cada indicador; no corresponden a una clasificación de inversión.")

            with st.expander("Evaluación de la silueta según el número de grupos"):
                valores_sil = []
                valores_k = list(range(2, min(6, len(base_km) - 1) + 1))
                for k in valores_k:
                    nuevos = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(x_km)
                    valores_sil.append(calcular_silueta(x_km, nuevos))
                fig_sil = px.line(x=valores_k, y=valores_sil, markers=True,
                    labels={"x": "Número de grupos (k)", "y": "Coeficiente de silueta"},
                    color_discrete_sequence=[AZUL_MEDIO])
                fig_sil.update_xaxes(tickmode="array", tickvals=valores_k)
                mostrar(estilo_grafico(fig_sil, alto=300))
            st.caption("K-Means se ajusta al año y municipios seleccionados. Es un ejercicio exploratorio independiente del agrupamiento del informe.")
        else:
            st.info("No hay suficientes municipios para probar K-Means con los filtros actuales.")


# TABLERO 8. RESULTADOS Y EXPLICACIÓN DEL MODELO GMM
#-------------------------------------------------------------------
with tab_gmm:
    st.markdown('<div class="seccion-pagina">Segmentación territorial mediante mezclas gaussianas (GMM)</div>',
        unsafe_allow_html=True)
    try:
        paquete_gmm = cargar_gmm(str(RUTA_GMM), RUTA_GMM.stat().st_mtime if RUTA_GMM.is_file() else None)
        configuracion_gmm = cargar_config_gmm(str(RUTA_GMM_CONFIG),
            RUTA_GMM_CONFIG.stat().st_mtime if RUTA_GMM_CONFIG.is_file() else None)
        evaluacion_gmm = cargar_evaluacion_gmm(str(RUTA_GMM_EVALUACION),
            RUTA_GMM_EVALUACION.stat().st_mtime if RUTA_GMM_EVALUACION.is_file() else None)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        st.error(f"No fue posible leer los resultados del GMM: {error}")
        paquete_gmm, configuracion_gmm, evaluacion_gmm = None, None, None

    if paquete_gmm is None:
        with st.container(border=True):
            st.info("Aún no existe una exportación del GMM. Ejecuta el script scripts/3_modelo_gmm.py "
                "para generar los archivos que utiliza esta pestaña.")
            st.code("python scripts/3_modelo_gmm.py", language="bash")
    else:
        gmm, claves, columnas_prob = paquete_gmm
        fusion = df_filtrado.merge(gmm, on=claves, how="inner", validate="one_to_one")
        if fusion.empty:
            st.warning("Los resultados del modelo no coinciden con los filtros actuales.")
        else:
            if configuracion_gmm:
                a, b, c, d = st.columns(4, gap="small")
                with a:
                    st.metric("Observaciones del modelo", numero(configuracion_gmm.get("observaciones", len(gmm))))
                with b:
                    st.metric("Grupos del modelo", numero(configuracion_gmm.get("grupos", len(gmm["cluster_gmm"].unique()))))
                with c:
                    st.metric("Coeficiente de silueta", numero(configuracion_gmm.get("silueta", np.nan), 3))
                with d:
                    st.metric("Varianza explicada por PCA", f"{numero(configuracion_gmm.get('varianza_pca2_pct', np.nan), 2)} %")

            # El mapa y las probabilidades muestran un solo año, evitando superponer tres puntos.
            if "anio" in claves:
                anios_modelo = sorted(fusion["anio"].unique().tolist())
                anio_gmm = st.selectbox("Año de la segmentación", anios_modelo,
                    index=len(anios_modelo) - 1, key="anio_gmm")
                fusion = fusion[fusion["anio"] == anio_gmm].copy()
            else:
                fusion = fusion.sort_values("anio").drop_duplicates("municipio", keep="last")

            grupos_posibles = sorted(fusion["cluster_gmm"].unique(),
                key=lambda valor: int(valor) if valor.isdigit() else valor)
            if configuracion_gmm and len(grupos_posibles) < configuracion_gmm.get("grupos", 0):
                st.caption("La segmentación se estimó con 16 municipios. Con la cobertura y año "
                    "seleccionados pueden aparecer menos grupos porque algunos corresponden a "
                    "territorios de referencia.")
            with st.container(border=True):
                grupos_sel = st.multiselect("Grupos que desea visualizar", grupos_posibles,
                    default=grupos_posibles, key="filtrar_gmm",
                    format_func=lambda valor: f"Grupo {valor}")
            fusion = fusion[fusion["cluster_gmm"].isin(grupos_sel)].copy()
            if fusion.empty:
                st.info("Selecciona al menos un grupo para consultar los resultados.")
            else:
                paleta_grupos = {"0": AZUL, "1": AZUL_MEDIO, "2": "#7997B8", "3": "#A1AEC0"}
                if columnas_prob:
                    fusion["probabilidad_maxima"] = fusion[columnas_prob].max(axis=1)
                col_proy, col_asig = st.columns([1.2, 1], gap="medium")

                with col_proy:
                    with st.container(border=True):
                        titulo_panel("Municipios según los grupos identificados")
                        if {"PC1", "PC2"}.issubset(fusion.columns):
                            fig_proy = px.scatter(fusion, x="PC1", y="PC2",
                                color="cluster_gmm", text="municipio", hover_name="municipio",
                                hover_data={"anio": True} if "anio" in claves else None,
                                color_discrete_map=paleta_grupos,
                                labels={"PC1": "Componente principal 1",
                                    "PC2": "Componente principal 2", "cluster_gmm": "Grupo"})
                            fig_proy.update_traces(textposition="top center", marker=dict(size=12))
                            mostrar(estilo_grafico(fig_proy, alto=440))
                        else:
                            resumen_grupos = fusion.groupby("cluster_gmm").size().reset_index(name="municipios")
                            fig_grupos = px.bar(resumen_grupos, x="cluster_gmm", y="municipios",
                                color="cluster_gmm", color_discrete_map=paleta_grupos,
                                text_auto=True, labels={"cluster_gmm": "Grupo", "municipios": "Municipios"})
                            fig_grupos.update_layout(showlegend=False)
                            mostrar(estilo_grafico(fig_grupos, alto=440))

                with col_asig:
                    with st.container(border=True):
                        titulo_panel("Clasificación de los municipios")
                        columnas = ["municipio", "cluster_gmm"]
                        if columnas_prob:
                            columnas.append("probabilidad_maxima")
                        tabla = fusion[columnas].sort_values(["cluster_gmm", "municipio"]).rename(columns={
                            "municipio": "Municipio", "cluster_gmm": "Grupo",
                            "probabilidad_maxima": "Probabilidad de pertenencia"})
                        if "Probabilidad de pertenencia" in tabla.columns:
                            tabla["Probabilidad de pertenencia"] = tabla["Probabilidad de pertenencia"].map(
                                lambda valor: f"{valor:.1%}")
                        tabla_datos(tabla, alto=440)

                if columnas_prob:
                    with st.container(border=True):
                        titulo_panel("Probabilidades de pertenencia a cada grupo")
                        probabilidades = fusion[["municipio", *columnas_prob]].copy()
                        probabilidades = probabilidades.rename(columns={
                            col: f"Grupo {col.rsplit('_', 1)[1]}" for col in columnas_prob})
                        largo = probabilidades.melt(id_vars="municipio", var_name="Grupo",
                            value_name="Probabilidad")
                        colores_prob = {f"Grupo {j}": paleta_grupos.get(str(j), GRIS_REFERENCIA)
                            for j in range(len(columnas_prob))}
                        fig_prob = px.bar(largo, x="municipio", y="Probabilidad", color="Grupo",
                            barmode="stack", color_discrete_map=colores_prob,
                            labels={"municipio": "Municipio"})
                        fig_prob.update_yaxes(tickformat=".0%", range=[0, 1])
                        mostrar(estilo_grafico(fig_prob, alto=400))
                        st.caption("Las probabilidades representan pertenencia estimada a los grupos, "
                            "no probabilidad de éxito de un proyecto de vivienda.")

                col_mapa, col_perfiles = st.columns([1.1, 1], gap="medium")
                with col_mapa:
                    with st.container(border=True):
                        titulo_panel("Distribución territorial de los grupos")
                        geo = fusion.dropna(subset=["lat", "lon"])
                        if geo.empty:
                            st.info("No hay ubicaciones disponibles para los municipios seleccionados.")
                        else:
                            centro, zoom = centro_mapa(geo)
                            fig_mapa = px.scatter_map(geo, lat="lat", lon="lon",
                                color="cluster_gmm", hover_name="municipio",
                                color_discrete_map=paleta_grupos, center=centro,
                                zoom=zoom, map_style="carto-positron",
                                labels={"cluster_gmm": "Grupo"})
                            fig_mapa.update_traces(marker=dict(size=14))
                            fig_mapa.update_layout(height=430, paper_bgcolor="#FFFFFF",
                                margin=dict(l=0, r=0, t=0, b=0), legend=dict(title="Grupo"))
                            mostrar(fig_mapa)
                            st.caption("Ubicaciones urbanas aproximadas; no son límites municipales.")
                with col_perfiles:
                    with st.container(border=True):
                        titulo_panel("Perfil de los grupos visibles")
                        variables_modelo = configuracion_gmm.get("variables", VARS_NORM) if configuracion_gmm else VARS_NORM
                        disponibles = [var for var in variables_modelo if var in fusion.columns]
                        if disponibles:
                            perfil = fusion.groupby("cluster_gmm")[disponibles].mean().T
                            perfil.index = [etiqueta(var) for var in perfil.index]
                            perfil.columns = [f"Grupo {valor}" for valor in perfil.columns]
                            tabla_datos(perfil.reset_index().rename(columns={"index": "Indicador"}).round(2), alto=430)
                            st.caption("Promedio de los indicadores en los municipios y el año seleccionados.")
                        else:
                            st.info("No hay indicadores disponibles para describir los grupos.")

                if evaluacion_gmm is not None:
                    with st.expander("Evaluación de grupos y trazabilidad del modelo"):
                        st.caption("Comparación de K-Means y GMM sobre los mismos dos componentes PCA. "
                            "El BIC y el AIC se interpretan solo dentro de GMM; valores menores son preferibles.")
                        columnas_eval = [col for col in ("metodo", "grupos", "silueta", "bic", "aic",
                            "grupos_efectivos", "convergencia") if col in evaluacion_gmm.columns]
                        tabla_datos(evaluacion_gmm[columnas_eval].round(3))
                        if configuracion_gmm:
                            st.caption("Modelo: GMM con covarianza full | Siete variables estandarizadas | "
                                "PCA con dos componentes | 48 observaciones municipio-año (10 del Valle de Aburrá y 6 de referencia). "
                                "La pestaña de PCA interactivo utiliza cinco indicadores, por lo que no es la misma configuración.")

                exportar_csv(fusion.drop(columns=["lat", "lon"], errors="ignore"),
                    "resultados_gmm_filtrados.csv")


# CRÉDITOS Y FUENTES
#-------------------------------------------------------------------
st.markdown("""
<div class="pie-equipo">ANTEPROYECTO ACADÉMICO | MIAD | UNIVERSIDAD DE LOS ANDES |
DATOS 2021–2023 | JOHAN VARGAS, RUBY ALVAREZ, ALEJANDRO DIAZ, DANEYRO CULCHAC</div>
""", unsafe_allow_html=True)
