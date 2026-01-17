# IA Delivery – Prototipo IA (Streamlit + Python)

Sistema de asignación de pedidos a camiones usando Random Forest, heurísticas de minimización de kilómetros y priorización por tiempo estimado. Incluye visualización interactiva con Streamlit y mapas con Folium/OpenRouteService.

---

## Requisitos

- Python 3.10+
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
git clone <TU_REPO_URL>
cd <TU_REPO_CARPETA>

pip install -r requirements.txt
# Nota: también puedes ejecutar todo en un solo comando (Linux/macOS):
# python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app/main_streamlit.py
