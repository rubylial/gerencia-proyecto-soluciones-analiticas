# ¿Dónde estudiar el próximo proyecto de vivienda?

Proyecto final de **Gerencia de Proyectos de Analytics** (MIAD, Universidad de los Andes).
Equipo: Juan Alejandro Díaz · Johan Vargas · Ruby Álvarez · Daneyro Culchac


Este proyecto junta las fuentes IGAC y DANE, 2021-2023, las vuelve comparables entre municipios y muestra los resultados en un **tablero interactivo**.

> Es una herramienta para **preseleccionar** municipios, no una recomendación de inversión.

## Qué hay en el repositorio

| Carpeta | Qué tiene |

| `datos/` | Los datos ya limpios y unificados |
| `analisis/` | Los 3 análisis del proyecto |
| `resultados/` | Las tablas y gráficos que generan los análisis |
| `dashboard/` | El tablero |
| `scripts/` | La limpieza inicial de los datos  |

## Cómo verlo en tu computador

**1. Descarga el proyecto e instala lo necesario** (una sola vez):

```
git clone https://github.com/rubylial/gerencia-proyecto-soluciones-analiticas.git
cd gerencia-proyecto-soluciones-analiticas
python -m venv .venv
.venv\Scripts\activate
pip install -r dashboard/requirements.txt
```

**2. Abre el tablero:**

```
streamlit run dashboard/app.py
```

Se abre solo en el navegador. Los resultados ya vienen incluidos en la carpeta `resultados/`, así que no hace falta correr nada más.


## Los 3 análisis

| Análisis | Qué responde |
|---|---|
| **Perfiles de municipios** (`3_pca_gmm.py`) | ¿Qué municipios se parecen entre sí? Los agrupa en 3 perfiles según su actividad inmobiliaria y de construcción. |
| **Preselección** (`4_multicriterio.py`) | ¿Cuáles conviene estudiar primero? Los ordena combinando compraventas, vivienda nueva y crecimiento de la población, y revisa si el orden se mantiene al cambiar la importancia de cada criterio. |
| **Valores atípicos** (`5_iqr_atipicos.py`) | ¿Qué compraventas tienen valores fuera de lo normal y deberían revisarse? |

## Para tener en cuenta

- El valor registrado en una compraventa **no es el precio de mercado**.
- Un valor atípico **no es un error**: es algo que vale la pena revisar.
- Los datos llegan hasta 2023 y son por municipio, no por barrio.

## Fuentes

- [IGAC – Registro de transacciones inmobiliarias](https://www.datos.gov.co/d/7y2j-43cv)
- [DANE – Proyecciones de población](https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion)
- [DANE – Licencias de construcción](https://www.dane.gov.co/index.php/estadisticas-por-tema/construccion/licencias-de-construccion)