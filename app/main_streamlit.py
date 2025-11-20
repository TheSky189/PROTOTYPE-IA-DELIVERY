# -*- coding: utf-8 -*-
# Interfaz web con Streamlit (prototipo completo con estado persistente + mapa)

import os, sys
import streamlit as st
import pandas as pd
import pydeck as pdk

# --- Resolver imports relativos desde la raíz del proyecto ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.preprocess import preparar_datos
from core.vrp_heuristic import planificar_rutas
from core.scoring import evaluar_solutions
from core.graph import ORIGEN_ID, ORIGEN_LATLON  # para el mapa
from core.vrp_savings import planificar_rutas_savings


# ==============================
#  Estado de sesión (persistencia)
# ==============================
if "pedidos" not in st.session_state: st.session_state.pedidos = None
if "productos" not in st.session_state: st.session_state.productos = None
if "destinos" not in st.session_state: st.session_state.destinos = None
if "agg" not in st.session_state: st.session_state.agg = None
if "ranking" not in st.session_state: st.session_state.ranking = None
if "best" not in st.session_state: st.session_state.best = None
if "soluciones" not in st.session_state: st.session_state.soluciones = None

# ==============================
#  Configuración de página
# ==============================
st.set_page_config(page_title="IA Delivery – Prototipo", layout="wide")
st.title("IA Delivery – Simulador de Rutas para Perecederos")

# ==============================
#  Sidebar: parámetros globales
# ==============================
st.sidebar.header("Parámetros de planificación")
capacidad_camion = st.sidebar.number_input("Capacidad por camión (unid.)", min_value=1, value=500, step=50)
vel_media = st.sidebar.number_input("Velocidad media (km/h)", min_value=1, value=70, step=5)
coste_km = st.sidebar.number_input("Coste por km (€)", min_value=0.0, value=0.8, step=0.1, format="%.2f")
max_camiones = st.sidebar.number_input("Nº máximo de camiones", min_value=1, value=10, step=1)

st.sidebar.header("Ponderación (scoring)")
alpha = st.sidebar.slider("α (peso km)", 0.0, 2.0, 1.0, 0.1)
beta  = st.sidebar.slider("β (peso nº vehículos)", 0.0, 2.0, 0.3, 0.1)
gamma = st.sidebar.slider("γ (penalización caducidad)", 0.0, 5.0, 2.0, 0.1)

# ==============================
#  1) Carga de datos
# ==============================
st.subheader("1) Cargar datos")

col1, col2, col3 = st.columns(3)
with col1:
    pedidos_file = st.file_uploader("Pedidos CSV", type=["csv"], key="pedidos_csv")
with col2:
    productos_file = st.file_uploader("Productos CSV", type=["csv"], key="productos_csv")
with col3:
    destinos_file = st.file_uploader("Destinos CSV", type=["csv"], key="destinos_csv")

c1, c2, c3 = st.columns([1,1,2])
with c1:
    if st.button("Usar datos de ejemplo"):
        # Lee los CSV demo desde /data del proyecto
        DATA_DIR = os.path.join(ROOT, "data")
        pedidos = pd.read_csv(os.path.join(DATA_DIR, "pedidos_demo.csv"))
        productos = pd.read_csv(os.path.join(DATA_DIR, "productos_demo.csv"))
        destinos = pd.read_csv(os.path.join(DATA_DIR, "destinos_demo.csv"))

        # Normalizar coordenadas a numérico
        for col in ["lat", "lon"]:
            if col in destinos.columns:
                destinos[col] = pd.to_numeric(destinos[col], errors="coerce")

        # Informe de coordenadas válidas
        if {"lat", "lon"}.issubset(destinos.columns):
            valid_coords = destinos["lat"].notna() & destinos["lon"].notna()
            st.info(f"Destinos con coordenadas válidas: {int(valid_coords.sum())} / {len(destinos)}")

        # Preparar datos
        agg, productos, destinos = preparar_datos(pedidos, productos, destinos)

        # Guardar en sesión
        st.session_state.pedidos = pedidos
        st.session_state.productos = productos
        st.session_state.destinos = destinos
        st.session_state.agg = agg
        st.session_state.ranking = None
        st.session_state.best = None
        st.session_state.soluciones = None

        st.success("Datos de ejemplo cargados.")
with c2:
    if st.button("Cargar desde ficheros"):
        if not (pedidos_file and productos_file and destinos_file):
            st.error("Sube los tres CSV (pedidos, productos y destinos).")
        else:
            try:
                pedidos = pd.read_csv(pedidos_file)
                productos = pd.read_csv(productos_file)
                destinos = pd.read_csv(destinos_file)

                # Normalizar coordenadas a numérico (si existen)
                for col in ["lat", "lon"]:
                    if col in destinos.columns:
                        destinos[col] = pd.to_numeric(destinos[col], errors="coerce")

                # (opcional) diagnóstico
                if {"lat", "lon"}.issubset(destinos.columns):
                    valid_coords = destinos["lat"].notna() & destinos["lon"].notna()
                    st.info(f"Destinos con coordenadas válidas: {int(valid_coords.sum())} / {len(destinos)}")

                # Preparar y guardar
                agg, productos, destinos = preparar_datos(pedidos, productos, destinos)

                st.session_state.pedidos = pedidos
                st.session_state.productos = productos
                st.session_state.destinos = destinos
                st.session_state.agg = agg
                st.session_state.ranking = None
                st.session_state.best = None
                st.session_state.soluciones = None

                st.success("Datos cargados desde ficheros.")
            except Exception as e:
                st.exception(e)

with c3:
    if st.button("Limpiar datos"):
        st.session_state.pedidos = None
        st.session_state.productos = None
        st.session_state.destinos = None
        st.session_state.agg = None
        st.session_state.ranking = None
        st.session_state.best = None
        st.session_state.soluciones = None
        st.info("Estado limpiado.")

# Vista previa rápida
if st.session_state.agg is not None:
    st.write("**Pedidos preparados (preview)**")
    st.dataframe(st.session_state.agg.head(), use_container_width=True)

# ==============================
#  2) Cálculo de planificación
# ==============================
st.subheader("2) Calcular planificación de rutas")
if st.button("Calcular mejor ruta"):
    if st.session_state.agg is None or st.session_state.destinos is None:
        st.error("Primero carga datos (demo o ficheros).")
    else:
        try:
            # Modelo A: heurístico v1
            soluciones_A = planificar_rutas(
                st.session_state.agg,
                st.session_state.productos,
                st.session_state.destinos,
                capacidad_camion=int(capacidad_camion),
                vel_media=float(vel_media),
                coste_km=float(coste_km),
                max_camiones=int(max_camiones),
            )
            # Modelo B: savings
            soluciones_B = planificar_rutas_savings(
                st.session_state.agg,
                st.session_state.destinos,
                capacidad_camion=int(capacidad_camion),
                vel_media=float(vel_media),
                coste_km=float(coste_km),
                max_camiones=int(max_camiones),
            )
            soluciones = soluciones_A + soluciones_B

            ranking = evaluar_solutions(soluciones, alpha=alpha, beta=beta, gamma=gamma)
            st.session_state.ranking = ranking
            st.session_state.soluciones = soluciones

            if not ranking.empty:
                best_id = ranking.iloc[0]["id"]
                best = next(s for s in soluciones if s["id"] == best_id)
                st.session_state.best = best
                st.success("Cálculo completado.")
            else:
                st.session_state.best = None
                st.warning("No se han generado soluciones.")
        except Exception as e:
            st.exception(e)
            st.session_state.best = None


# ==============================
#  3) Resultados (persistentes)
# ==============================
st.subheader("3) Resultados")
if st.session_state.ranking is not None:
    st.write("**Ranking de soluciones (multi-modelo)**")
    st.dataframe(st.session_state.ranking, use_container_width=True)

    # Vista compacta A vs B
    if st.session_state.soluciones:
        df_comp = pd.DataFrame([{
            "Modelo": s["id"],
            "Km totales": s["kpi"]["km_totales"],
            "Vehículos": s["kpi"]["vehiculos"],
            "Tiempo total (h)": s["kpi"]["tiempo_total_h"],
            "Caducidad OK": "✅" if s["kpi"].get("caducidad_ok", True) else "❌",
        } for s in st.session_state.soluciones])
        st.write("**Comparativa rápida (A/B)**")
        st.dataframe(df_comp, use_container_width=True)

# Mejor solución + descargas
if st.session_state.best is not None:
    best = st.session_state.best

    st.subheader("Mejor solución")
    # KPIs
    kpi = best["kpi"]
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Km totales", f"{kpi['km_totales']:.1f}")
    colB.metric("Vehículos", kpi["vehiculos"])
    colC.metric("Tiempo total (h)", f"{kpi['tiempo_total_h']:.1f}")
    colD.metric("Caducidad OK", "✅" if kpi.get("caducidad_ok", True) else "❌")

    # Tabla de rutas
    import json, io
    df_rutas = pd.DataFrame([{
        "Vehículo": r["vehiculo"],
        "Secuencia": " → ".join(r["secuencia"]),
        "Carga": r["carga_total"],
        "Km": r["km"],
        "Tiempo (h)": r["tiempo_h"],
        "Coste (€)": r["coste"],
        "Caducidad OK": "✅" if r["caducidad_ok"] else "❌",
        "Pedidos": ", ".join(r["pedidos"]) if r["pedidos"] else "—",
    } for r in best["rutas"]])
    st.write("**Rutas planificadas**")
    st.dataframe(df_rutas, use_container_width=True)

    # Descargas
    buf_csv = io.StringIO()
    df_rutas.to_csv(buf_csv, index=False)
    st.download_button("Descargar rutas (CSV)", buf_csv.getvalue(),
                       file_name="rutas_mejor_solucion.csv", mime="text/csv")
    st.download_button("Descargar solución (JSON)",
                       json.dumps(best, ensure_ascii=False, indent=2),
                       file_name="mejor_solucion.json", mime="application/json")


# ==============================
#  Mapa de rutas (Pydeck) robusto
# ==============================
st.subheader("Mapa de rutas")

# 1) Comprobar que haya datos y solución
if st.session_state.destinos is None:
    st.info("No hay 'destinos' cargados todavía. Carga datos y calcula una planificación.")
    st.stop()

if st.session_state.best is None:
    st.info("No hay una solución seleccionada aún. Calcula la planificación para ver el mapa.")
    st.stop()

# 2) Copia + normalización segura de destinos
destinos = st.session_state.destinos.copy()
for col in ["lat", "lon"]:
    if col not in destinos.columns:
        destinos[col] = pd.NA
    destinos[col] = pd.to_numeric(destinos[col], errors="coerce")

# 3) Filtrado de coordenadas válidas
valid_mask = destinos["lat"].notna() & destinos["lon"].notna()
destinos_valid = destinos.loc[valid_mask].copy()
faltan = len(destinos) - len(destinos_valid)
if faltan > 0:
    st.warning(f"{faltan} destino(s) sin coordenadas válidas: no aparecerán en el mapa.")

# 4) Función auxiliar para obtener coordenadas por id de nodo

def _coord_de_nodo(node_id: str):
    if node_id == ORIGEN_ID:
        lat, lon = ORIGEN_LATLON
        return [float(lon), float(lat)]  # pydeck: [lon, lat]
    if "id_destino" not in destinos_valid.columns:
        return None
    row = destinos_valid.loc[destinos_valid["id_destino"] == node_id]
    if row.empty:
        return None
    return [float(row.iloc[0]["lon"]), float(row.iloc[0]["lat"])]

# 5) Construcción de paths con color por vehículo
palette = [
    [33, 158, 188], [251, 133, 0], [42, 157, 143],
    [233, 196, 106], [244, 162, 97], [231, 111, 81],
]
paths, centros = [], []
for i, r in enumerate(st.session_state.best["rutas"]):
    coords = []
    for node in r["secuencia"]:
        c = _coord_de_nodo(node)
        if c:
            coords.append(c)
            centros.append(c)
    if len(coords) >= 2:
        paths.append({"vehiculo": f"Vehículo {r['vehiculo']}", "path": coords, "color": palette[i % len(palette)]})

# 6) Puntos (origen + destinos de la mejor ruta)
puntos = [{"name": ORIGEN_ID, "coord": [float(ORIGEN_LATLON[1]), float(ORIGEN_LATLON[0])]}]
for r in st.session_state.best["rutas"]:
    for node in r["secuencia"]:
        if node == ORIGEN_ID:
            continue
        c = _coord_de_nodo(node)
        if c:
            puntos.append({"name": node, "coord": c})

# 7) Capas de Pydeck
layers = []
if paths:
    layers.append(pdk.Layer("PathLayer", data=paths, get_path="path", get_width=4,
                            get_color="color", pickable=True))
if puntos:
    layers.append(pdk.Layer("ScatterplotLayer", data=puntos, get_position="coord",
                            get_radius=5000, pickable=True))

# 8) Vista inicial
if centros:
    avg_lon = sum(p[0] for p in centros) / len(centros)
    avg_lat = sum(p[1] for p in centros) / len(centros)
else:
    avg_lat, avg_lon = ORIGEN_LATLON

view_state = pdk.ViewState(latitude=float(avg_lat), longitude=float(avg_lon), zoom=6)

st.pydeck_chart(pdk.Deck(
    map_style=None,  # si quieres fondo Mapbox, define MAPBOX_API_KEY y cambia el estilo
    initial_view_state=view_state,
    layers=layers,
    tooltip={"text": "{vehiculo}"}
))

# 9) Diagnóstico rápido del mapa
with st.expander("Diagnóstico del mapa"):
    st.write(f"Paths construidos: {len(paths)}")
    st.write(f"Puntos dibujados: {len(puntos)}")
    if len(paths) == 0:
        st.info("No hay polilíneas. Revisa que 'id_destino' en rutas exista en 'destinos' y tenga lat/lon.")
