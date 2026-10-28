import os
import json
import warnings

import gradio as gr
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
from libpysal.weights import KNN

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 2026
VARIABLES = ["desempleo", "crecimiento", "luces_nocturnas"]
VARIABLES_NOMBRES = {
    "desempleo": "Desempleo",
    "crecimiento": "Crecimiento económico",
    "luces_nocturnas": "Luces nocturnas",
}
URL_GEOJSON = (
    "https://raw.githubusercontent.com/"
    "caticoa3/colombia_mapa/master/"
    "co_2018_MGN_DPTO_POLITICO.geojson"
)


def cargar_mapa():
    """Carga el mapa departamental y estandariza el campo de nombre."""
    mapa = gpd.read_file(URL_GEOJSON)
    mapa = mapa.loc[mapa.geometry.notna() & ~mapa.geometry.is_empty].copy()

    candidatos = [
        "departamento", "NOMBRE_DPT", "DPTO_CNMBR", "NOMBRE",
        "NAME_1", "NAME", "nom_dpto", "nombre_dpt", "DPTO"
    ]
    campo_nombre = next((c for c in candidatos if c in mapa.columns), None)

    if campo_nombre is None:
        posibles = [
            c for c in mapa.columns
            if c != mapa.geometry.name and mapa[c].dtype == "object"
            and mapa[c].nunique(dropna=True) >= 25
        ]
        if not posibles:
            raise ValueError(
                "No se encontró el campo con los nombres de los departamentos "
                f"en el GeoJSON. Columnas disponibles: {list(mapa.columns)}"
            )
        campo_nombre = posibles[0]

    mapa["departamento"] = mapa[campo_nombre].astype(str).str.strip()
    mapa = mapa.drop_duplicates(subset="departamento").reset_index(drop=True)
    if len(mapa) < 25:
        raise ValueError(f"Se esperaban departamentos de Colombia; se encontraron {len(mapa)} registros.")
    return mapa


def crear_pesos_knn(mapa, k=4):
    """Construye pesos KNN usando coordenadas proyectadas en metros."""
    proyectado = mapa.to_crs(epsg=3116)
    pesos = KNN.from_dataframe(proyectado, k=min(k, len(proyectado) - 1))
    pesos.transform = "R"
    return pesos, pesos.sparse.toarray()


def generar_panel_sintetico(mapa, W, anios=range(2015, 2025), semilla=SEED):
    """Genera datos sintéticos reproducibles para probar la aplicación."""
    rng = np.random.default_rng(semilla)
    n = len(mapa)
    datos = []

    desempleo = np.clip(rng.normal(11.0, 3.0, n), 3.0, 25.0)
    crecimiento = rng.normal(3.0, 1.8, n)
    luces = np.clip(rng.lognormal(2.5, 0.55, n), 1.0, 35.0)

    for anio in anios:
        if anio != min(anios):
            desempleo = np.clip(
                0.68 * desempleo + 0.08 * (W @ desempleo)
                + 2.2 + rng.normal(0, 1.2, n), 2.0, 30.0
            )
            crecimiento = (
                0.35 * crecimiento + 0.10 * (W @ crecimiento)
                + 1.8 + rng.normal(0, 1.4, n)
            )
            luces = np.clip(
                0.80 * luces + 0.08 * (W @ luces)
                + 4.0 + rng.normal(0, 2.0, n), 0.5, 60.0
            )

        for i, nombre in enumerate(mapa["departamento"]):
            datos.append({
                "departamento": nombre,
                "anio": int(anio),
                "desempleo": float(desempleo[i]),
                "crecimiento": float(crecimiento[i]),
                "luces_nocturnas": float(luces[i]),
                "origen_datos": "SINTÉTICOS"
            })

    return pd.DataFrame(datos)


def estimar_modelo(panel, mapa, W):
    """Estimación exploratoria: Y_t = c + A Y_(t-1) + B W Y_(t-1) + e."""
    anios = sorted(panel["anio"].unique())
    n = len(mapa)
    k = len(VARIABLES)
    Y_por_anio = {}

    for anio in anios:
        datos = (
            panel.loc[panel["anio"] == anio]
            .set_index("departamento")
            .reindex(mapa["departamento"])[VARIABLES]
        )
        Y_por_anio[anio] = datos.to_numpy(dtype=float)
        if not np.isfinite(Y_por_anio[anio]).all():
            raise ValueError(f"Hay valores ausentes o no finitos en el año {anio}.")

    X_lista, Y_lista = [], []
    for t in range(1, len(anios)):
        anterior = Y_por_anio[anios[t - 1]]
        actual = Y_por_anio[anios[t]]
        X_t = np.column_stack([np.ones(n), anterior, W @ anterior])
        X_lista.append(X_t)
        Y_lista.append(actual)

    X = np.vstack(X_lista)
    Y_objetivo = np.vstack(Y_lista)
    coeficientes, _, rango, _ = np.linalg.lstsq(X, Y_objetivo, rcond=None)
    A = coeficientes[1:1 + k, :].T
    B = coeficientes[1 + k:1 + 2 * k, :].T

    # Diagnóstico dinámico preliminar de la representación estimada.
    F = np.kron(np.eye(n), A.T) + np.kron(W, B.T)
    radio_espectral = float(np.max(np.abs(np.linalg.eigvals(F))))
    estable = bool(np.isfinite(radio_espectral) and radio_espectral < 1)

    errores = Y_objetivo - X @ coeficientes
    tabla_ajuste = pd.DataFrame({
        "variable": VARIABLES,
        "RMSE": np.sqrt(np.mean(errores ** 2, axis=0)),
        "MAE": np.mean(np.abs(errores), axis=0),
    })

    return A, B, radio_espectral, estable, tabla_ajuste, len(Y_objetivo), int(rango)


# Preparar datos y estimación al iniciar la aplicación.
gdf = cargar_mapa()
w_knn, Wk = crear_pesos_knn(gdf, k=4)
anios = list(range(2015, 2025))
panel = generar_panel_sintetico(gdf, Wk, anios=anios)
A, B, radio_espectral, estable, tabla_ajuste, n_obs, rango_diseno = estimar_modelo(panel, gdf, Wk)


def crear_mapa(variable, anio):
    anio = int(anio)
    datos_anio = (
        panel.loc[panel["anio"] == anio, ["departamento", variable]]
        .drop_duplicates("departamento")
    )
    mapa = gdf.merge(datos_anio, on="departamento", how="left")
    geojson = json.loads(mapa.to_json())

    figura = px.choropleth(
        mapa,
        geojson=geojson,
        locations="departamento",
        featureidkey="properties.departamento",
        color=variable,
        hover_name="departamento",
        color_continuous_scale="Viridis",
        projection="mercator",
        title=f"{VARIABLES_NOMBRES[variable]} por departamento — {anio}",
        labels={variable: VARIABLES_NOMBRES[variable]},
    )
    figura.update_geos(fitbounds="locations", visible=False)
    figura.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=600)
    return figura


def interpretar_resultados():
    estado = (
        "cumple el criterio preliminar de estabilidad"
        if estable else "no cumple el criterio preliminar de estabilidad"
    )
    return f"""
### Interpretación automática

- **Tipo de ejercicio:** modelo dinámico espacial exploratorio.
- **Variables:** desempleo, crecimiento económico y luces nocturnas.
- **Unidades territoriales:** {len(gdf)} departamentos.
- **Periodo:** {min(anios)}–{max(anios)}.
- **Matriz de pesos:** K vecinos más cercanos (KNN), con k = 4.
- **Observaciones utilizadas en la estimación:** {n_obs}.
- **Radio espectral calculado:** {radio_espectral:.4f}.
- **Auditoría:** el sistema {estado} según el criterio preliminar de radio espectral menor que 1.

La matriz A resume las asociaciones dinámicas locales y la matriz B resume las asociaciones espaciales rezagadas.

**Advertencia metodológica:** los datos son sintéticos y se usan para probar la aplicación. No representan estimaciones oficiales de los departamentos de Colombia. Los coeficientes no deben interpretarse como efectos causales.

**Alcance:** esta versión es exploratoria; no estima todavía todos los mecanismos del SpVAR solicitados en el taller ni sus impactos directos, indirectos y totales.
"""


def crear_tabla(matriz):
    return pd.DataFrame(matriz, index=VARIABLES, columns=VARIABLES).rename_axis("Variable actual").reset_index().round(4)


with gr.Blocks(theme=gr.themes.Soft(), title="Econometría Espacial — SpVAR") as demo:
    gr.Markdown("""
    # Laboratorio de Econometría Espacial
    ### Análisis exploratorio con un modelo dinámico espacial (SpVAR)

    Aplicación demostrativa para explorar indicadores territoriales, dinámica temporal y asociaciones espaciales.
    """)

    with gr.Row():
        variable_sel = gr.Dropdown(
            choices=list(VARIABLES_NOMBRES.keys()), value="desempleo",
            label="Indicador para el mapa"
        )
        anio_sel = gr.Dropdown(
            choices=anios, value=max(anios), label="Año"
        )

    mapa_salida = gr.Plot(label="Mapa departamental")
    actualizar_mapa = gr.Button("Actualizar mapa", variant="primary")
    actualizar_mapa.click(fn=crear_mapa, inputs=[variable_sel, anio_sel], outputs=mapa_salida)

    gr.Markdown("---\n## Resultados del modelo")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Matriz A: dinámica local")
            gr.Dataframe(value=crear_tabla(A), interactive=False)
        with gr.Column():
            gr.Markdown("### Matriz B: dinámica espacial rezagada")
            gr.Dataframe(value=crear_tabla(B), interactive=False)

    gr.Markdown("## Diagnóstico del ajuste")
    gr.Dataframe(value=tabla_ajuste.round(4), interactive=False)
    gr.Markdown("## Auditoría de estabilidad")
    gr.Markdown(
        f"**Radio espectral:** {radio_espectral:.6f}  \n"
        f"**Criterio radio espectral < 1:** {'Cumplido' if estable else 'No cumplido'}"
    )
    gr.Markdown(interpretar_resultados())
    gr.Markdown("""
    ---
    **Nota:** todos los indicadores de esta versión son sintéticos. La aplicación necesita una conexión a internet al iniciar para descargar el GeoJSON.
    """)

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=puerto)
