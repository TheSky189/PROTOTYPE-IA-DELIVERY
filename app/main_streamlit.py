# -*- coding: utf-8 -*-
# Interfaz web con Streamlit

import streamlit as st
import pandas as pd
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.preprocess import preparar_datos
from core.vrp_heuristic import planificar_rutas
from core.scoring import evaluar_solutions

st.set_page_config(page_title="IA Delivery – Prototipo", layout="wide")
st.title("IA Delivery – Simulador de Rutas para Perecederos")

# --- Sidebar: parámetros globales ---
st.sidebar.header("Parámetros")
capacidad_camion = st.sidebar.number_input("Capacidad por camión (unidades)", min_value=1, value=500)
vel_media = st.sidebar.number_input("Velocidad media (km/h)", min_value=1, value=70)
coste_km = st.sidebar.number_input("Coste por km (€)", min_value=0.0, value=0.8)
max_camiones = st.sidebar.number_input("Nº máximo de camiones", min_value=1, value=10)
alpha = st.sidebar.slider("α (peso km)", 0.0, 2.0, 1.0, 0.1)
beta = st.sidebar.slider("β (peso nº vehículos)", 0.0, 2.0, 0.3, 0.1)
gamma = st.sidebar.slider("γ (penalización caducidad)", 0.0, 5.0, 2.0, 0.1)

# --- Carga de datos ---
st.subheader("1) Cargar datos")
col1, col2, col3 = st.columns(3)
with col1:
    pedidos_file = st.file_uploader("Pedidos CSV", type=["csv"])
with col2:
    productos_file = st.file_uploader("Productos CSV", type=["csv"])
with col3:
    destinos_file = st.file_uploader("Destinos CSV", type=["csv"])

if st.button("Usar datos de ejemplo"):
    DATA_DIR = os.path.join(ROOT, "data")
    pedidos = pd.read_csv(os.path.join(DATA_DIR, "pedidos_demo.csv"))
    productos = pd.read_csv(os.path.join(DATA_DIR, "productos_demo.csv"))
    destinos = pd.read_csv(os.path.join(DATA_DIR, "destinos_demo.csv"))

else:
    pedidos = pd.read_csv(pedidos_file) if pedidos_file else None
    productos = pd.read_csv(productos_file) if productos_file else None
    destinos = pd.read_csv(destinos_file) if destinos_file else None

if all([pedidos is not None, productos is not None, destinos is not None]):
    st.success("Datos cargados correctamente.")
    st.write("**Pedidos**", pedidos.head())
    st.write("**Productos**", productos.head())
    st.write("**Destinos**", destinos.head())

    # --- Preparación de datos ---
    st.subheader("2) Preparar y validar datos")
    df_pedidos, df_productos, df_destinos = preparar_datos(pedidos, productos, destinos)
    st.write("Pedidos preparados:")
    st.dataframe(df_pedidos.head())

    # --- Cálculo ---
    st.subheader("3) Calcular mejor planificación de rutas")
    if st.button("Calcular mejor ruta"):
        soluciones = planificar_rutas(
            df_pedidos, df_productos, df_destinos,
            capacidad_camion=capacidad_camion,
            vel_media=vel_media,
            coste_km=coste_km,
            max_camiones=max_camiones,
        )
        ranking = evaluar_solutions(soluciones, alpha=alpha, beta=beta, gamma=gamma)
        st.success("Cálculo completado.")
        st.write("**Ranking de soluciones**:")
        st.dataframe(ranking)

        # Mostrar la mejor solución con detalle
        if not ranking.empty:
            best_id = ranking.iloc[0]["id"]
            best = next(s for s in soluciones if s["id"] == best_id)
            st.subheader("4) Mejor solución")
            st.json(best)
else:
    st.info("Sube los tres CSV o pulsa 'Usar datos de ejemplo'.")