# IA Delivery – Prototipo IA (Streamlit + Python)

Sistema de asignación de pedidos a camiones usando Random Forest, heurísticas de minimización de kilómetros y priorización por tiempo estimado. Incluye visualización interactiva con Streamlit y mapas con Folium/OpenRouteService.

---

## Requisitos

- Python 3.10+
- Git
- Conexión a Internet (para ORS y descargas de librerías)
- CSVs de datos de ejemplo:
  - `destinos_geocodificados.csv`
  - `pedidos.csv`
  - `lineasPedido.csv`
  - `productos.csv`

---

## Instalación y configuración paso a paso

1. **Clonar el repositorio**

```bash
git clone https://github.com/TheSky189/PROTOTYPE-IA-DELIVERY
cd <TU_REPO_CARPETA>

pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app/main_streamlit.py
