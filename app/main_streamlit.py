import streamlit as st

st.set_page_config(page_title="Manual de uso - Maqueta", layout="wide")
st.title("Asignación de Pedidos a Camiones (Maqueta de uso)")

st.sidebar.subheader("Configuración de las variables")

capacidad_camion = st.sidebar.number_input(
    "Capacidad del camión (unidades)",
    min_value=1,
    max_value=5000,
    value=500,
    step=1
)
st.sidebar.caption("Número máximo de unidades que puede transportar un camión.")

velocidad_media = st.sidebar.number_input(
    "Velocidad media (km/h)",
    min_value=5.0,
    max_value=200.0,
    value=80.0,
    step=1.0
)
st.sidebar.caption("Velocidad media estimada de los camiones.")

precio_litro = st.sidebar.number_input(
    "Precio combustible (€/L)",
    min_value=0.1,
    max_value=5.0,
    value=1.65,
    step=0.01
)
st.sidebar.caption("Precio del litro de combustible usado en los cálculos (maqueta visual).")

st.sidebar.number_input(
    "Consumo estimado (L / 100 km)",
    value=round(velocidad_media * 25.0 / 80.0, 2),
    disabled=True
)
st.sidebar.caption("Consumo estimado calculado automáticamente según velocidad (solo visual).")

st.header("1) Subir datasets (4 CSV obligatorios)")

col1, col2, col3, col4 = st.columns(4)

uploaded = st.session_state.get("uploaded_files", [False, False, False, False])

with col1:
    if st.button("Subir Destinos.csv"):
        uploaded[0] = True
    st.write("Subido" if uploaded[0] else "No subido")

with col2:
    if st.button("Subir Pedidos.csv"):
        uploaded[1] = True
    st.write("Subido" if uploaded[1] else "No subido")

with col3:
    if st.button("Subir LineasPedido.csv"):
        uploaded[2] = True
    st.write("Subido" if uploaded[2] else "No subido")

with col4:
    if st.button("Subir Productos.csv"):
        uploaded[3] = True
    st.write("Subido" if uploaded[3] else "No subido")

st.session_state["uploaded_files"] = uploaded

if all(uploaded):
    st.success("Todos los CSV 'subidos'. ¡Puedes continuar al siguiente paso!")
else:
    st.info("Sube los 4 CSV para continuar (solo maqueta visual).")

st.header("2) Parámetros para el cálculo (maqueta)")

p1, p2 = st.columns(2)
with p1:
    n_clusters = st.slider(
        "Número de clusters geográficos (zonas)",
        min_value=1,
        max_value=20,
        value=6
    )
    st.caption("Divide los pedidos en zonas geográficas (solo visual).")

with p2:
    rf_estimators = st.number_input(
        "Random Forest estimators",
        min_value=10,
        max_value=1000,
        value=200,
        step=10
    )
    st.caption("Número de árboles del modelo RF (solo maqueta).")


st.header("3) Entrenar Random Forest y asignar pedidos (maqueta)")

if all(uploaded):
    if st.button("Entrenar y asignar"):
        st.info("Simulación: modelo entrenado y pedidos asignados (solo maqueta).")
else:
    st.button("Entrenar y asignar", disabled=True)

st.markdown("---")

st.header("4) Resumen (maqueta visual)")

st.subheader("Resumen coste combustible por camión")
st.dataframe(
    {
        "Camión": [1,2,3],
        "Km": [123.45, 98.76, 150.0],
        "Coste (€)": [25.0, 20.0, 30.0],
        "Pedidos": [10, 8, 12]
    }
)

st.success("Km total (todos los camiones): 372.21 km")
st.success("Coste total combustible (todos los camiones): 75.0 €")

st.header("5) Resultado de asignación (maqueta)")

for i in range(1,4):
    with st.expander(f"Camión {i} — pedidos: {i*4} — carga total: {i*50}", expanded=False):
        st.dataframe(
            {
                "PedidoID": [f"P{i}{j}" for j in range(1,5)],
                "Ciudad": ["Ciudad A","Ciudad B","Ciudad C","Ciudad D"],
                "Cantidad_total": [10, 20, 15, 5],
                "TiempoTotalEstimado": [30, 25, 40, 20],
                "ClusterID": [1,1,2,2]
            }
        )

st.header("6) Mostrar ruta por camión (maqueta)")

for i in range(1,4):
    if st.button(f"Mostrar ruta Camión {i}"):
        st.info(f"Visualización de mapa para Camión {i} (solo maqueta, sin ruta real).")
